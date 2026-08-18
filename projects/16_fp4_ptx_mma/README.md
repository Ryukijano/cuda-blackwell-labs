# Project 16 — Hand-rolled FP4 PTX MMA

A minimal warp-synchronous FP4 Tensor Core MMA using the native Blackwell PTX instruction:

```
.mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
  .f32.e2m1.e2m1.f32.ue4m3
```

## Build & run

```bash
cd projects/16_fp4_ptx_mma
make all
make run
```

The `Makefile` uses the alternate-stage architecture pair `compute_121a`/`sm_121a`; the plain `sm_121` ptxas target does **not** accept block-scaled `mma.sync`.

## What it does

- One warp per block.
- Every lane packs identical `e2m1` values (`0x2` = 1.0) into four A and two B 32-bit registers.
- Block scales are set to `0x38` (`ue4m3` = 1.0).
- Calls the inline-PTX mma.sync and writes the first four FP32 accumulators.
- A correct `m16n8k64` of all-1.0 operands returns `64.0` in each accumulator.

## Notes

- This is a **probe / building block**, not a tiled FP4 GEMM.
- A full GEMM requires per-lane fragment packing, shared-memory or register tiling, and the correct `ue4m3` scale distribution from the PTX ISA.
