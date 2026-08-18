# Project 18: Key Findings

- On hardware-coherent UMA (GB10), `cudaMallocManaged` can be prefetched to the GPU and the same physical LPDDR5X is measured.
- `st.global.cs` forces writes toward DRAM rather than just updating L2, giving a more honest write-bandwidth number.
- `ld.global.cg` bypasses L1 and is useful for true memory-bound bandwidth measurement.
- Results can be compared with `bandwidth_test` (Project 02) to see the effect of cache operators.
