# Project 15 — CUTLASS 3.8 Hello GEMM

A minimal FP16 GEMM using the CUTLASS 3.8 device-level `Gemm` API.

## Build & run

```bash
cd projects/15_cutlass_gemm
make fetch    # clones CUTLASS 3.8 into ./cutlass (ignored by git)
make all
make run
make clean
```

## What it does

- Builds an `M×N×K` column-major FP16 GEMM via `cutlass::gemm::device::Gemm`.
- Fills A/B/C using CUTLASS random-fill utilities.
- Runs the CUTLASS kernel and compares against `cutlass::reference::host::Gemm`.
- Benchmarks 11 iterations and reports median time and effective TFLOPS.

## Notes

- This uses CUTLASS 3.8's high-level device API, not the lower-level CuTe DSL.
- The default configuration targets `cutlass::arch::Sm70` / `OpClassSimt`, which works on GB10 but does not push the Tensor Cores to their peak. A follow-up would explicitly instantiate `OpClassTensorOp` with `GemmShape<...>` tiles tuned for `Sm80/Sm90`.
