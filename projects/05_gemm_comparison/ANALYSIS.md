# Project 05: Five-Way GEMM Comparison — Analysis

## Results at 4096x4096 (FP32)

| Implementation | Time (ms) | TFLOP/s | BW (GB/s) | Speedup vs Naive |
|----------------|-----------|---------|-----------|------------------|
| naive | 112.6 | 1.22 | 1.8 | 1.0x |
| tiled_16 | 87.8 | 1.57 | 2.3 | 1.3x |
| tiled_32 | 78.2 | 1.76 | 2.6 | 1.4x |
| vec4 | 103.4 | 1.33 | 1.9 | 1.1x |
| tiled_reg | 51.1 | 2.69 | 3.9 | 2.2x |
| **cuBLAS** | **7.7** | **17.90** | **26.2** | **14.7x** |

## Results at 1024x1024 (fits in L2 cache)

| Implementation | Time (ms) | TFLOP/s | BW (GB/s) | Speedup |
|----------------|-----------|---------|-----------|---------|
| naive | 1.23 | 1.74 | 10.2 | 1.0x |
| tiled_16 | 0.91 | 2.35 | 13.8 | 1.3x |
| tiled_reg | 0.53 | 4.05 | 23.7 | 2.3x |
| **cuBLAS** | **0.14** | **15.64** | **91.6** | **9.0x** |

## Why Each Implementation Is Slow

### 1. Naive GEMM (1.22 TFLOP/s)
- Each thread reads an entire row of A (K=4096 floats = 16KB) and an entire column of B
- No data reuse: A is read N times, B is read M times
- Total global memory reads: M*K*N + K*N*M = 2*M*K*N = 2*4096^3 = 137 billion floats
- At 273 GB/s peak, this takes: 137e9 * 4 / 273e9 = ~2 seconds (we see 113ms because of L2 caching)
- **Bottleneck: Global memory bandwidth. No data reuse.**

### 2. Tiled 16x16 (1.57 TFLOP/s)
- Each tile of 16x16 is loaded into shared memory once and reused 16 times
- Reduces global memory access by 16x compared to naive
- But 16x16 = 256 threads per block, only 6 blocks per SM (limited by warps)
- Shared memory: 2 * 16*16 * 4 = 2KB per block (tiny, not the limiter)
- **Bottleneck: Small tile size → low arithmetic intensity, still memory-bound**

### 3. Tiled 32x32 (1.76 TFLOP/s)
- 32x32 tile = 1024 threads per block → only 1 block per SM (1536/1024 = 1 warp group)
- Better data reuse (32x) but lower occupancy (1 block vs 6 for 16x16)
- **Bottleneck: Low occupancy (1 block/SM) limits latency hiding**

### 4. Vectorized float4 (1.33 TFLOP/s)
- float4 loads from A help, but B is accessed column-wise (strided, not coalesced)
- The float4 load for A is wasted because B's access pattern is the bottleneck
- **Bottleneck: B's column access is non-coalesced — float4 on A alone doesn't help**

### 5. Tiled + Register Accumulation (2.69 TFLOP/s)
- 64x64 block, each thread computes 4 outputs (TM=4)
- Better register reuse: each loaded B value is used 4 times
- 1024 threads per block → 1 block per SM, but 4x more work per thread
- **Bottleneck: Still memory-bound, but 2x better arithmetic intensity than tiled_16**

### 6. cuBLAS (17.90 TFLOP/s)
- **14.7x faster than naive, 6.6x faster than tiled_reg**
- Uses TF32 Tensor Cores (even for FP32 input, cuBLAS may use TF32 internally)
- Highly optimized tiling: typically 128x128 or 256x128 blocks
- Uses warp-level matrix multiply-accumulate (MMA) instructions
- Pipeline overlapping: loads next tile while computing current
- Register-tile: each warp computes a 16x16x16 tile using Tensor Cores
- **Bottleneck: Compute-bound at Tensor Core throughput, not memory-bound**

## The 14.7x Gap: What cuBLAS Does That We Don't

1. **Tensor Cores**: cuBLAS uses WMMA/MMA instructions that compute 16x16x16 matrix
   multiply in one instruction. Our kernels use scalar FFMA (one multiply-add per instruction).
   Tensor Cores provide ~8x throughput for FP32 (TF32 mode).

2. **Optimal tiling**: cuBLAS uses 128x128 or larger tiles with multi-level tiling
   (register tiles → shared memory tiles → global memory). Our 16x16 and 32x32 tiles
   are too small for good arithmetic intensity.

3. **Pipeline overlapping**: cuBLAS overlaps memory loads with computation using
   asynchronous memory operations (`cp.async` or `LDGSTS`). Our kernels serialize
   load → compute → load → compute.

4. **Warp-level coordination**: cuBLAS uses warp shuffles to share data between threads
   in a warp, avoiding shared memory bank conflicts. Our kernels use shared memory
   with potential bank conflicts.

5. **Auto-tuning**: cuBLAS selects the best kernel variant based on matrix dimensions
   and GPU architecture. Our kernels are fixed-size.

## Scaling Behavior

| Size | Naive TFLOP/s | cuBLAS TFLOP/s | Gap |
|------|---------------|----------------|-----|
| 128 | 0.65 | 0.36 | 0.6x (cuBLAS overhead dominates) |
| 256 | 1.16 | 2.12 | 1.8x |
| 512 | 1.61 | 8.10 | 5.0x |
| 1024 | 1.74 | 15.64 | 9.0x |
| 2048 | 1.64 | 16.84 | 10.3x |
| 4096 | 1.22 | 17.90 | 14.7x |

- **Small matrices (128)**: cuBLAS is slower due to launch overhead and kernel selection overhead
- **Medium (512-1024)**: cuBLAS pulls ahead as Tensor Cores become utilized
- **Large (4096)**: cuBLAS reaches 17.9 TFLOP/s, naive drops to 1.22 (L2 cache exhausted)

## Key Takeaways

1. **Naive GEMM is memory-bound**: 1.22 TFLOP/s at 4096x4096, limited by global memory bandwidth
2. **Tiling helps but not enough**: 1.3-2.2x speedup, still memory-bound
3. **cuBLAS is 14.7x faster**: Uses Tensor Cores + optimal tiling + pipeline overlapping
4. **Tensor Cores are the key**: They provide 8x+ throughput for matrix multiply
5. **Small matrices favor simplicity**: cuBLAS overhead makes it slower at 128x128
6. **The GB10's 273 GB/s bandwidth is the ceiling for non-Tensor-Core GEMM**

## CUTLASS and PyTorch Notes

CUTLASS and PyTorch are not installed on this system. To add them:

```bash
# CUTLASS (from source)
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=121
make -j

# PyTorch (with CUDA support)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CUTLASS would provide a GEMM that's closer to cuBLAS performance (within 90-95%)
because it uses the same techniques (Tensor Cores, tiling, pipelining) but with
user-configurable templates.

PyTorch's `torch.matmul` dispatches to cuBLAS internally, so it would match cuBLAS
performance (within 1-2% for dispatch overhead).
