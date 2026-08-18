# Project 20: UMA Fault / Residency Latency Probe

Measures UMA access latency in three residency states:

| Pass | Meaning |
|------|---------|
| COLD | CPU touches all pages first, then the GPU reads them (may fault) |
| WARM | Pages are prefetched to the GPU with `cudaMemPrefetchAsync` |
| PRESSURE | Half CPU-resident, half GPU-resident (contention / thrash) |

The kernel uses `ld.global.cv.f32` (cache-volatile, bypasses L1/L2) and `clock64()` for cycle-accurate measurement.

## Build and run

```bash
make
make run
```
