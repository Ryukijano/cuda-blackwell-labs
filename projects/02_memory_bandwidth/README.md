# Project 2 — Memory Bandwidth & Latency Lab

This directory contains the implementation of the second CUDA Blackwell Labs project.

## Files

| File | Purpose |
|------|---------|
| `bandwidth_test.cu` | CUDA memory benchmark suite (9 kernels + sweep harness) |
| `bandwidth_simple.cu` | Legacy simple memory-bandwidth microbenchmark |
| `deep_dive.cu` | Legacy UMA and allocation-type deep dive |
| `plot_results.py` | Generates graphs and tables from CSV output |
| `Makefile` | Build, test, run, profile, and verify |
| `protocol.md` | Experiment protocol |
| `README.md` | This file |
| `results/` | CSVs, generated PNGs, and tables |

## Build and run

```bash
make all
make test      # CPU reference correctness checks
make run       # full sweep + plots
make ncu       # Nsight Compute profile for 3 kernels (requires perf counter permissions)
make verify    # checks that all expected artifacts exist
make clean     # remove binary, logs, and plots (keeps CSVs)
```

Compile command (also used by the Makefile):

```bash
nvcc -arch=sm_121 -lineinfo -o bandwidth_test bandwidth_test.cu
```

## Kernels

- `sequential_read`
- `sequential_write`
- `read_write_copy`
- `saxpy`
- `strided_read`
- `random_read`
- `coalesced_access`
- `non_coalesced_access`
- `atomic_accumulate`

## Security design

- No `popen()`, `system()`, or shell execution in the benchmark binary.
- All command-line arguments are validated against an allow-list.
- All `printf`/`fprintf`/`log_printf` format strings are literals.
- `__attribute__((format(printf, 1, 2)))` is used on `log_printf`.
- All allocation sizes are overflow-checked before `cudaMalloc`/`cudaHostAlloc`.
- CPU reference tests run before any timed sweep.

## Results (DGX Spark, GB10, 273 GB/s peak)

Selected measured values from a single run:

| Kernel / Condition | Size | Bandwidth (GB/s) | % peak |
|---|---|---|---|
| `sequential_read` (baseline) | 256 MB | ~210 | ~77% |
| `sequential_write` | 256 MB | ~195 | ~71% |
| `read_write_copy` | 256 MB | ~208 | ~76% |
| `saxpy` | 256 MB | ~220 | ~80% |
| `strided_read` stride=8 | 64 MB | ~53 | ~19% |
| `random_read` | 256 MB | ~27 | ~10% |
| `non_coalesced_access` | 256 MB | ~140 | ~51% |
| `atomic_accumulate` | 64 MB | ~5.8 | ~2% |
| `sequential_read` block=32 | 64 MB | ~100 | ~37% |
| `sequential_read` block=1024 | 64 MB | ~216 | ~79% |
| `cudaMalloc` (device) | 256 MB | ~209 | ~77% |
| `cudaMallocManaged` first touch | 256 MB | ~164 | ~60% |
| `cudaMallocManaged` warm | 256 MB | ~165 | ~60% |
| `cudaHostAlloc` | 256 MB | ~170 | ~62% |
| `cudaMalloc` + CPU contention | 256 MB | ~179 | ~66% |
| `cudaMallocManaged` + CPU contention | 256 MB | ~161 | ~59% |

Full CSV data and generated plots are in `results/`.

### Nsight Compute

The `make ncu` target is configured and the binary supports `--ncu <kernel>`. On this workstation, `ncu` reports `ERR_NVGPUCTRPERM` unless the user has root or `cap_perfmon` access. The target runs with `make -k` semantics so the build does not abort; a short CSV with the error is produced in `results/ncu_*.csv`.

## Output files

- `results/bandwidth_workset.csv` — working-set sweep
- `results/bandwidth_stride.csv` — stride sweep
- `results/bandwidth_block.csv` — block-size sweep
- `results/bandwidth_alloc.csv` — allocation-type sweep
- `results/bandwidth_uma.csv` — UMA contention
- `results/table_alloc.md` — allocation-type table
- `results/table_uma.md` — UMA contention table
- `results/bandwidth_*.png` — generated plots
- `results/ncu_*.csv` — Nsight Compute reports (when profiling is permitted)
