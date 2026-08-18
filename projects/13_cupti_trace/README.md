# Project 13 — CUPTI Activity Trace

A minimal CUPTI activity-trace program that records kernel launches, timings, grid/block sizes, and memory copies. Unlike the CUPTI range profiler, this does not require root-level profiling permissions (`CAP_SYS_ADMIN`) and can be run as a normal user.

## Build & run

```bash
cd projects/13_cupti_trace
make all
make run
make clean
```

## What it does

- Registers `bufferRequested`/`bufferCompleted` callbacks with CUPTI.
- Enables `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`, `CUPTI_ACTIVITY_KIND_MEMCPY`, and `CUPTI_ACTIVITY_KIND_RUNTIME`.
- Launches a sample `warmup_kernel` several times.
- Flushes the CUPTI buffers and prints a trace of each kernel: name, duration, grid, and block.

## Why?

CUPTI is the API behind tools like `nsys` and `ncu`. The activity API is the easiest way to see exactly which kernels launched and how long they took, without adding manual `cudaEventRecord` timers. It also reveals launch dimensions that might be different from what you expected.

## Note on counters

The CUPTI *Range Profiler* (event/metric counters) requires elevated permissions on this machine and fails with `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`. The activity trace avoids that path entirely.
