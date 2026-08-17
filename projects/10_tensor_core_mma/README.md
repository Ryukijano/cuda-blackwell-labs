# Project 10 — Hand-rolled FP16 Tensor Core MMA

A from-scratch, warp-level WMMA FP16 GEMM that drives the Tensor Cores directly on the GB10 (SM121), then compares against cuBLAS on the same data.

## Build & run

```bash
cd projects/10_tensor_core_mma
make all      # build mma
make run      # compare WMMA hand-rolled kernel vs cuBLAS
make ptx      # generate mma.ptx
make profile  # Nsight Compute profile
make clean
```

## What it does

- Each warp uses `nvcuda::wmma` to compute one 16x16 tile of `C`.
- A and B are kept in FP16; the accumulator is also FP16.
- A is stored row-major (M x K), B is stored column-major (K x N), and C is row-major (M x N).
- The hand-rolled kernel is intentionally simple (no shared memory, no double buffering) so the cuBLAS comparison is educational.

## Sizes tested

| Size | cuBLAS (ms) | cuBLAS TFLOP/s | WMMA (ms) | WMMA TFLOP/s |
|------|-------------|------------------|-----------|--------------|
| 256³ | ~0.008 | ~4.3 | ~0.007 | ~5.1 |
| 512³ | ~0.012 | ~22.9 | ~0.025 | ~10.8 |
| 1024³ | ~0.038 | ~56.0 | ~0.156 | ~13.8 |
| 2048³ | ~0.206 | ~83.3 | ~1.234 | ~13.9 |

*(Sanitizer run timings are much slower and should not be used for performance comparisons.)*

## Key takeaways

- The Tensor Cores can be accessed directly via `nvcuda::wmma` without cuBLAS.
- A naïve WMMA kernel is **much slower** than cuBLAS (10–20× at large sizes) because cuBLAS uses tiling, shared memory, pipelining, and register-level optimizations.
- For production GEMMs, prefer cuBLAS/cuBLASLt; use WMMA for custom kernels where a library does not fit.
