# Project 22: Key Findings

- CuTe provides a composable, layout-driven DSL for writing GEMM kernels without hand-rolling every index.
- The simple FMA-based kernel here is portable and runs on `sm_121`, but it is not using Tensor Cores.
- The `cp_async_fence()/cp_async_wait<0()` sequence is included for compatibility with `cp.async` tiled copies; on this simplified kernel it is mostly a no-op because the default copy path uses explicit loads/stores.
- 512³ FP16 GEMM achieves ~5.6 TFLOPS on the GB10 (well below Tensor Core peaks, as expected for an FMA-only kernel).
- The next step is a CuTe `TiledMMA` with `MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>` (or the SM120 block-scaled atoms for NVFP4).
