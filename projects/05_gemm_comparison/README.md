# Project 05 — Five-Way GEMM Comparison

Compare six matrix-multiplication implementations from naive CUDA to cuBLAS on the GB10.

## Quick start

```bash
cd projects/05_gemm_comparison
make all         # build gemm_test
make test        # quick correctness check (128–512)
make run         # full benchmark (128–4096)
make ptx         # generate PTX
make profile     # Nsight Compute profile (requires perf counters)
make clean
```

## Implementations

| Kernel | Description |
|---|---|
| `naive` | One thread per output, no shared memory |
| `tiled_16` | 16×16 shared-memory tile |
| `tiled_32` | 32×32 shared-memory tile |
| `vec4` | `float4` loads for A, scalar for B |
| `tiled_reg` | 64×64 tile, each thread computes 4 outputs |
| `cuBLAS` | cuBLAS `cublasSgemm` (FP32) |

## Artifacts

- `gemm_test` — benchmark executable (ignored by git)
- `gemm.ptx` — generated PTX
- `../../results/05_gemm_comparison.txt` — full benchmark output
- `ANALYSIS.md` — annotated findings
