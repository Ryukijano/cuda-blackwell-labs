# Project 15: CUTLASS 3.8 Hello GEMM — Analysis

## Results

| Metric | Value |
|--------|-------|
| Problem | 1024 × 1024 × 1024 FP16 |
| Time | 0.137 ms |
| TFLOPS | ~15.7 |
| Verify | PASSED |

## Observations

- The `cutlass::gemm::device::Gemm` default for FP16 here uses the SIMT path (`OpClassSimt`), not the Tensor Core path.
- 15.7 TFLOPS is far below cuBLAS/Tensor Core peak, confirming that the default tile/arch configuration is not extracting the tensor cores.

## Takeaways

1. CUTLASS 3.8 compiles cleanly on `sm_121` with the relaxed-constexpr flag.
2. To reach higher TFLOPS, the next step is to explicitly use `cutlass::arch::OpClassTensorOp`, `cutlass::arch::Sm80`/`Sm90`, and a tuned `GemmShape`/`WarpShape`/`InstructionShape`.
3. The `cutlass::HostTensor` and reference utilities make correctness verification straightforward.
