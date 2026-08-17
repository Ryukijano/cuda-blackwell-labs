# Experiment Protocol — Memory Bandwidth & Latency Lab

## Objective
Implement a suite of CUDA memory access kernels and measure effective bandwidth across access patterns, working-set sizes, strides, block sizes, and allocation types on the GB10 DGX Spark.

## Primary outcome
Effective memory bandwidth (GB/s) for a `sequential_read` kernel at a 256 MB working set with `cudaMalloc`, used as the baseline reference point.

## Secondary outcomes (exploratory)
- Effective bandwidth for all 9 kernels across the full working-set sweep.
- Stride vs bandwidth curve for `strided_read` at 64 MB.
- Block size vs bandwidth curve for `sequential_read` at 64 MB.
- Bandwidth across `cudaMalloc`, `cudaMallocManaged`, and `cudaHostAlloc`.
- UMA CPU-contention degradation measurement.
- Nsight Compute metric profile for 3 selected kernels.

## Conditions
| ID | Description |
|----|-------------|
| baseline | `sequential_read`, 256 MB, `cudaMalloc`, block 256, default stream |
| small_wset | 1 KB – 256 MB working-set sweep |
| large_wset | 512 MB – 64 GB working-set sweep (memory permitting) |
| stride | `strided_read` at 64 MB with strides 1, 2, 4, 8, 16, 32, 64, 128 |
| block | `sequential_read` at 64 MB with blocks 32, 64, 128, 256, 512, 1024 |
| alloc | `sequential_read` at 256 MB with `cudaMalloc`, `cudaMallocManaged`, `cudaHostAlloc` |
| uma | `sequential_read` at 256 MB with GPU-only and GPU+CPU memory contention |
| ncu | `sequential_read`, `read_write_copy`, `saxpy` profiled with Nsight Compute |

## Controls and ablations
- Each kernel result is compared against a CPU reference on a small, fixed test vector.
- Bandwidth is calculated as `bytes_moved / kernel_time_seconds` using CUDA events.
- Warm-up runs precede timed runs; at least 3 timed iterations are averaged.
- Random index buffers for `random_read` are generated with a fixed seed.
- CPU contention is a single pinned memory writer running on the host during the GPU kernel.

## Data
- Generated in-memory buffers of the specified working-set sizes.
- No external datasets.

## Hardware
- DGX Spark (GB10, SM121, 128 GB LPDDR5X UMA).
- CUDA 13.0, driver 580.142.

## Analysis plan
- Plot bandwidth vs working-set size, stride, and block size.
- Tabulate allocation-type and UMA-contention results.
- Identify plateau, 50% stride, CPU-contention loss, and sustained-vs-peak ratio.
- No p-hacking; all sweeps pre-registered above.

## Stopping rules
- Stop if any kernel fails CPU reference.
- Stop if working-set allocation fails.
- Stop if Nsight Compute returns no counters.

## Pre-registration status
- [x] Protocol written before execution.
- [x] Security design reviewed (no user input, bounded buffers, literal format strings, no shell in binary).
- [x] CPU reference tests planned for all kernels.
- [x] Build, run, verify, and ncu targets defined.
