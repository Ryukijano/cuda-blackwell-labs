# Project 10: Hand-rolled FP16 Tensor Core MMA — Analysis

## Goal

Show how to call the Tensor Cores directly from a CUDA kernel and understand how much performance a hand-rolled implementation leaves on the table compared with a vendor library.

## Implementation

- Uses `nvcuda::wmma` and `wmma::mma_sync` to issue Tensor Core MMA instructions.
- Each warp computes a 16×16 tile of `C`.
- A is M×K row-major, B is K×N column-major, C is M×N row-major.
- The kernel is deliberately simple: no shared memory, no double buffering, no inter-warp tiling.

## Why it is slow vs cuBLAS

1. **Global memory loads per fragment.** Every `wmma::load_matrix_sync` reads directly from global memory. cuBLAS stages tiles through shared memory and reuses them across many warps.
2. **One tile per warp.** Each warp owns exactly one 16×16 tile. cuBLAS uses larger blocks and multiple accumulators per warp to hide latency.
3. **No pipeline/tiling.** The kernel loads the next tile only after the current MMA finishes. cuBLAS pipelines loads and MMAs.
4. **No vectorization/tuning.** cuBLAS is tuned for the exact SM, memory, and problem sizes.

## Results

| Size | cuBLAS (ms) | cuBLAS TFLOP/s | WMMA (ms) | WMMA TFLOP/s |
|------|-------------|------------------|-----------|--------------|
| 256³ | 0.008 | 4.3 | 0.007 | 5.1 |
| 512³ | 0.012 | 22.9 | 0.025 | 10.8 |
| 1024³ | 0.038 | 56.0 | 0.156 | 13.8 |
| 2048³ | 0.206 | 83.3 | 1.234 | 13.9 |

- cuBLAS reaches ~83 TFLOP/s at 2048³, while the naïve WMMA kernel plateaus near ~14 TFLOP/s.
- The WMMA kernel is a useful learning tool but is **not** a replacement for cuBLAS.

## Takeaway

Use the Tensor Core PTX/WMMA path when you need a custom operation that cuBLAS cannot express. For standard GEMMs, cuBLAS/cuBLASLt is almost always faster.
