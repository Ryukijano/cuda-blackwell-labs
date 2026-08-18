# Project 11: TMA 2D Tile Copy — Analysis

## Results

| Implementation | 4096×4096 int copy | Effective BW |
|----------------|--------------------|--------------|
| TMA            | ~0.59 ms           | ~226 GB/s    |
| Naive thread copy | ~0.61 ms        | ~220 GB/s    |

- Both paths are near the GB10's peak DRAM bandwidth (~273 GB/s).
- The TMA path is a faithful minimal demo; it is not significantly faster than the naive path because the kernel is not instruction- or index-calculation-bound at 64MB.
- The value of TMA becomes clearer in larger kernels (deep GEMM pipelines, 3D stencils, etc.) where the same tensor descriptor can be reused and the hardware walks complex layouts while warps compute.

## Takeaways

1. TMA requires a `CUtensorMap` created on the host and passed via `__grid_constant__`.
2. TMA loads need an mbarrier; TMA stores use the `bulk_group` completion mechanism.
3. Shared memory for TMA must be 128-byte aligned.
4. The `cp.async.bulk.tensor` instructions are available on `sm_121a` and compile with PTX 9.0.

## What would make it faster

- Pipelined TMA loads with double-buffered shared memory.
- Swizzled shared layouts to avoid bank conflicts.
- Larger tile sizes (e.g. 128×128) and more CTAs.
- `L2::cache_hint` hints for `evict_first`/`evict_last`.
