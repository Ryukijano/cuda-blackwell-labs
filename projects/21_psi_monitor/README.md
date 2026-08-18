# Project 21: PSI / Memory Stall Monitor

Samples `/proc/pressure/memory` while a GPU kernel hammers unified memory.

## Why PSI?

On GB10, direct UVM telemetry is unavailable:
- Nsight Systems UVM tracing is unsupported.
- CUPTI UVM events are structurally absent.
- NVML memory clock is not exposed.

`/proc/pressure/memory` reports the percentage of time the kernel spends stalled on memory. The `some` line means *some* task is stalled; the `full` line means *all* non-idle tasks are stalled.

## Build and run

```bash
make
make run
```
