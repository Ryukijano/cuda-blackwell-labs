# Project 06 — Precision Lab

Benchmark GEMM, pointwise ops, reduction, and softmax across FP32, TF32, FP16, and BF16 on the GB10 SM121.  An optional narrow-precision add-on (`precision_narrow`) exercises FP8 E4M3 and NVFP4 E2M1 via cuBLASLt.

## Quick start

```bash
cd projects/06_precision_lab
make all         # build precision_test
make narrow      # build and run FP8 / NVFP4 check
make run         # run standard benchmarks
make ptx         # generate PTX
make profile     # Nsight Compute profile
make clean
```

## Benchmarks

| Section | What it measures |
|---|---|
| GEMM (cuBLAS) | FP32, TF32, FP16, BF16 at 512–4096 |
| Narrow GEMM (cuBLASLt) | FP8 E4M3, NVFP4 E2M1 at 512–4096 |
| Pointwise ops | ReLU, SiLU, GELU bandwidth (FP32) |
| Reduction | FP32 sum reduction bandwidth |
| Softmax | FP32 softmax at 1024×1024 and 4096×4096 |

## Findings

- FP8 E4M3 is fully supported on the GB10 with cuBLASLt and reaches ~150 TFLOP/s at 4096².
- NVFP4 E2M1 requires VEC16_UE4M3 block-scaling and a 128×4 swizzled scale layout; the installed CUDA 13.0 / cuBLAS 13.1.1 stack reports `UNSUPPORTED` for simple scalar-scale configurations.  To go further, use the official `LtNvfp4Matmul` sample or CUTLASS `79_blackwell_geforce_gemm`.

## Artifacts

- `precision_test` — standard benchmark executable (ignored by git)
- `precision_narrow` — FP8 / NVFP4 check executable (ignored by git)
- `precision.ptx` — generated PTX
- `../../results/06_precision_lab.txt` — standard output
- `../../results/06_precision_lab_narrow.txt` — narrow-precision output
- `ANALYSIS.md` — annotated findings
