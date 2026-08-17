# Project 06 — Precision Lab

Benchmark GEMM, pointwise ops, reduction, and softmax across FP32, TF32, FP16, and BF16 on the GB10 SM121.

## Quick start

```bash
cd projects/06_precision_lab
make all         # build precision_test
make run         # run all benchmarks
make ptx         # generate PTX
make profile     # Nsight Compute profile
make clean
```

## Benchmarks

| Section | What it measures |
|---|---|
| GEMM (cuBLAS) | FP32, TF32, FP16, BF16 at 512–4096 |
| Pointwise ops | ReLU, SiLU, GELU bandwidth (FP32) |
| Reduction | FP32 sum reduction bandwidth |
| Softmax | FP32 softmax at 1024×1024 and 4096×4096 |

## Artifacts

- `precision_test` — benchmark executable (ignored by git)
- `precision.ptx` — generated PTX
- `../../results/06_precision_lab.txt` — benchmark output
- `ANALYSIS.md` — annotated findings
