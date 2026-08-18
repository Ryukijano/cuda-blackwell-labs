# Project 16: Hand-rolled FP4 PTX MMA — Analysis

## Results

| Metric | Value |
|--------|-------|
| Instruction | `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` |
| A/B value | `0x2` (e2m1 = 1.0) |
| Scale value | `0x38` (ue4m3 = 1.0) |
| Accumulators (lane 0) | 64.0, 64.0, 64.0, 64.0 |
| Status | Executed successfully |

## Observations

- The FP4 warp-synchronous `mma.sync` **does** assemble and execute on `sm_121a` with the current CUDA 13.0 toolkit.
- `compute_121a`/`sm_121a` is required; the default `sm_121` ptxas target rejects the `.kind::mxf4nvf4` modifier.
- The output `64.0` matches the expected dot product of 64 K values of 1.0, so the data-to-register encoding chosen is bit-correct.

## Takeaways

1. Consumer Blackwell (GB10, SM121) can execute native FP4 Tensor Core `mma.sync` today.
2. The tooling constraint is the PTX assembler target, not the driver — the kernel loads and runs without `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` once assembled for `sm_121a`.
3. A real FP4 GEMM is the next step: tiling, scale-factor memory layout, and epilogue are the remaining work.
