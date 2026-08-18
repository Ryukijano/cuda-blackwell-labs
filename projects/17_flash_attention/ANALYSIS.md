# Project 17: FlashAttention-style Online Softmax Attention — Analysis

## Results

| Metric | Value |
|--------|-------|
| N | 256 |
| D | 64 |
| Time | 0.27 ms |
| Max err vs CPU | 0.000000 |
| Verify | PASSED |

## Observations

- The online-softmax update produces numerically identical output to the two-pass CPU softmax.
- Because one thread serialises over all keys, the TFLOPS is low, but the algorithmic idea is clearly demonstrated.
- The output matrix is produced in one pass with only `O(N * D)` extra per-query state (`m`, `l`, `o[0..D]`).

## Takeaways

1. The FlashAttention "online softmax" trick works: you can compute softmax attention without storing `N x N` attention scores.
2. The kernel is memory-bound; tiling `K` and `V` through shared memory is the next optimisation.
3. A full tiled implementation would also fuse the `QK^T` and `PV` loops and use warp-level reductions for the dot products.
