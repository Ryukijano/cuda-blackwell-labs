# Project 22: CuTe DSL Hello GEMM

A minimal, self-contained CuTe 3.8 tensor-programming example.

This is the real CuTe DSL counterpart to Project 15 (CUTLASS 3.8 `device::Gemm`).
The kernel tiles the matrices into shared memory and computes `C = A * B` using
`cute::gemm` and FMA in registers. It runs on `sm_121` and does **not** require
TMEM or the SM100 `tcgen05` path.

## Build and run

```bash
make
make run
make run M=1024 K=1024
```

If CUTLASS has not been fetched yet, the Makefile will run `make fetch` in
`projects/15_cutlass_gemm/`.
