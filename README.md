# CUDA Blackwell Labs

A 9-project learning plan for CUDA mastery on the **NVIDIA DGX Spark (GB10 Grace Blackwell, SM121)**.

## Hardware (from Project 01 probe)

| Property | Value |
|----------|-------|
| GPU | NVIDIA GB10 |
| Compute capability | 12.1 (SM121) |
| SMs | 48 |
| CUDA cores | 6,144 (128/SM) |
| Tensor Cores | ~384 (5th gen, FP4/FP6/FP8/BF16/FP16/TF32) |
| Max threads/SM | 1,536 (48 warps) |
| Max blocks/SM | 24 |
| Shared mem/block (opt-in) | 99 KB |
| Shared mem/SM | 100 KB |
| Registers/SM | 65,536 |
| L2 cache | 24 MB |
| Memory | 128 GB LPDDR5X unified (CPU+GPU) |
| Memory data rate | 8,533 MT/s |
| Memory bus | 256 bits |
| Peak bandwidth | 273 GB/s |
| CUDA | 13.0 |
| Driver | 13.0 |

### Critical GB10 facts
- **128 GB is NOT HBM.** It is shared LPDDR5X. Bandwidth (273 GB/s) is far below datacenter GPUs.
- **No TMEM, no WGMMA, no DSMEM.** Not all datacenter Blackwell features exist on SM121.
- **Unified memory:** `cudaMemGetInfo()` underreports. Always cross-check with `/proc/meminfo`.
- **Compile with `-arch=sm_121 -lineinfo`** for GB10-specific SASS and profiling support.

## Key Findings Summary

### Memory (Project 02)
- **Sustained bandwidth**: 231 GB/s reads (85% of 273 peak), 198 GB/s writes (73%)
- **L2 cache boundary**: 24 MB — bandwidth drops from ~900 GB/s (L2) to ~231 GB/s (DRAM) at 26+ MB
- **Stride 2** causes 50% bandwidth drop
- **Random reads**: 40 GB/s (15% peak) — 6x slower than sequential
- **UMA contention**: 15% bandwidth loss when CPU hammers memory concurrently
- **Managed memory**: 30% slower than cudaMalloc even with cudaMemAdvise + prefetch
- **Latency**: ~19 ns (L1), ~140 ns (L2), ~148 ns (DRAM)

### PTX/SASS (Project 03)
- Compiler uses **FMA** (fused multiply-add) automatically
- **Uniform registers** (UR4+) on SM121 for block-uniform values (blockIdx, n, a)
- **Descriptor-based addressing** (`desc[UR4]`) on SM90+, not direct `[R]`
- `-Xptxas -O3` controls device code optimization (not `-O3`)
- **float4** loads produce `LDG.E.128` (128-bit vectorized) instructions

### Occupancy (Project 04)
- **Memory access pattern** matters more than occupancy (6x difference)
- **Shared memory > 48KB** drops occupancy to 16%, starts to hurt performance
- **Block size 64-512** is optimal; 32 and 1024 are suboptimal
- **Warp divergence** has minimal effect on memory-bound kernels
- **Compiler optimizes away register pressure** — all variants used only 11 actual registers

### GEMM (Project 05)
- **cuBLAS is 14.7x faster** than naive at 4096x4096 (17.9 vs 1.22 TFLOP/s)
- **Tiling helps 1.3-2.2x** but still memory-bound without Tensor Cores
- **Tensor Cores** are the key differentiator (8x+ throughput)
- **GB10 is memory-bound for non-Tensor-Core GEMM**

### Precision (Project 06)
- **FP16/BF16 GEMM**: ~91 TFLOP/s (5x faster than FP32 at 18 TFLOP/s)
- **TF32 GEMM**: ~41 TFLOP/s (2.3x faster than FP32, same memory footprint)
- **Pointwise ops** (ReLU, GELU, SiLU): ~230 GB/s — memory-bound, precision doesn't affect compute
- **Reduction**: 225 GB/s — memory-bound
- **Softmax**: 215 GB/s for large N (memory-bound), faster for small N (L2 cached)

### Streams & Async (Project 07)
- **Default stream + pageable memory**: 473.7 ms for 16 batches (29.6 ms/batch)
- **Multiple streams**: 3.5 ms total — **135x speedup**
- **Pinned + non-blocking streams**: 2.87 ms — **165x speedup**
- **Stream-ordered allocation**: 11.2 ms — lower peak memory via pool reuse
- **UMA copy lesson**: `cudaMemPrefetchAsync` is **25x faster** than `cudaMemcpy` (page table op vs real copy)

### CUDA Graphs (Project 08)
- Eager stream (20 tiny kernels × 1000 iters): 83.6 ms
- CUDA Graph (same): 44.9 ms — **1.86x speedup**
- Best for: many small, static-shape kernels where CPU launch overhead dominates

### NVDEC (Project 09)
- CPU decode: ~186-224 FPS for 1080p30
- NVDEC + CPU preprocess: 130 FPS (NV12→RGB CPU conversion is the bottleneck)
- NVDEC only wins when frames stay GPU-resident for cvcuda preprocess

## Project List

| # | Project | Phase | Status |
|---|---------|-------|--------|
| 1 | GB10 Hardware Probe | 1 — Hardware & Memory | ✅ Complete |
| 2 | Memory Bandwidth & Latency Lab | 1 | ✅ Complete (+ deep dive) |
| 3 | CUDA → PTX → SASS Pipeline | 1 | ✅ Complete |
| 4 | Occupancy & Stall Experiments | 2 — Compiler & SM | ✅ Complete |
| 5 | Five-Way GEMM Comparison | 2 | ✅ Complete |
| 6 | Precision Lab (FP32→FP4) | 2 | ✅ Complete |
| 7 | Streams, Events, Async Allocation | 3 — Runtime & Systems | ✅ Complete |
| 8 | CUDA Graphs | 3 | ✅ Complete |
| 9 | NVDEC Video Pipeline | 3 | ✅ Complete |

## Directory Structure

```
cuda-blackwell-labs/
├── .gitignore                  # Build artifacts, binaries, result files
├── Makefile                    # Top-level: make, make test, make profile
├── README.md                   # This file
├── common/
│   ├── cuda_utils.h            # Error macros, timers, device props, memory info
│   └── benchmark.h             # Benchmark runner, statistics, comparison tables
├── projects/
│   ├── 01_hardware_probe/      # ✅ Complete
│   │   ├── Makefile
│   │   ├── probe.cu
│   │   └── probe.ptx
│   ├── 02_memory_bandwidth/    # ✅ Complete
│   │   ├── Makefile
│   │   ├── bandwidth.cu
│   │   ├── deep_dive.cu
│   │   └── ANALYSIS.md
│   ├── 03_ptx_sass/            # ✅ Complete
│   │   ├── Makefile
│   │   ├── pipeline.cu
│   │   ├── ANALYSIS.md
│   │   └── sass_*.txt
│   ├── 04_occupancy_stalls/    # ✅ Complete
│   │   ├── Makefile
│   │   ├── occupancy.cu
│   │   └── ANALYSIS.md
│   ├── 05_gemm_comparison/     # ✅ Complete
│   │   ├── Makefile
│   │   ├── gemm.cu
│   │   └── ANALYSIS.md
│   ├── 06_precision_lab/       # ✅ Complete
│   │   ├── Makefile
│   │   ├── precision.cu
│   │   └── ANALYSIS.md
│   ├── 07_streams_events/      # ✅ Complete
│   │   ├── Makefile
│   │   ├── streams.cu
│   │   └── ANALYSIS.md
│   ├── 08_cuda_graphs/         # ✅ Complete
│   │   ├── Makefile
│   │   ├── graphs.cu
│   │   └── ANALYSIS.md
│   └── 09_nvdec_pipeline/      # ✅ Complete
│       ├── Makefile
│       ├── nvdec_pipeline.py
│       └── ANALYSIS.md
└── results/                    # Generated benchmark outputs (in repo for reference)
```

## Build & Run

```bash
# Build all projects
make

# Build and run a specific project
cd projects/01_hardware_probe
make run

# Run deep dive (Project 02)
cd projects/02_memory_bandwidth
make deep

# Generate SASS (Project 03)
cd projects/03_ptx_sass
make sass
make arch_compare

# Profile with Nsight Compute
cd projects/05_gemm_comparison
make profile

# Run Python-based NVDEC pipeline
cd projects/09_nvdec_pipeline
make
```

## Requirements

- NVIDIA DGX Spark or other SM121 Blackwell GPU (or `-arch=sm_121` can be changed)
- CUDA 13.0+
- GCC 11+
- `make`
- For Project 09: conda environment `3d_recon` with PyTorch, OpenCV, cvcuda

## Precision Performance Summary (4096x4096 GEMM, cuBLAS)

| Precision | Time (ms) | TFLOP/s | Speedup vs FP32 | Memory |
|-----------|-----------|---------|-----------------|--------|
| FP32 | 7.54 | 18.2 | 1.0x | 201 MB |
| TF32 | 3.31 | 41.5 | 2.3x | 201 MB (same) |
| FP16 | 1.51 | 91.3 | 5.0x | 101 MB (0.5x) |
| BF16 | 1.51 | 90.9 | 5.0x | 101 MB (0.5x) |

## GEMM Performance Summary (4096x4096, FP32)

| Implementation | Time (ms) | TFLOP/s | Speedup vs Naive |
|----------------|-----------|---------|------------------|
| naive | 112.6 | 1.22 | 1.0x |
| tiled_16 | 87.8 | 1.57 | 1.3x |
| tiled_32 | 78.2 | 1.76 | 1.4x |
| vec4 | 103.4 | 1.33 | 1.1x |
| tiled_reg | 51.1 | 2.69 | 2.2x |
| **cuBLAS** | **7.7** | **17.9** | **14.7x** |
