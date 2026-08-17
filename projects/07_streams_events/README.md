# Project 07 — Streams, Events, Async Allocation

Explores concurrency and memory models on the GB10 (Grace-Blackwell UMA):

1. **Default stream** — synchronous, single queue baseline.
2. **Multiple streams** — inter-batch parallelism with `cudaStreamCreate`.
3. **Pinned + non-blocking streams** — page-locked host memory and `cudaStreamNonBlocking`.
4. **Stream-ordered allocation** — `cudaMallocAsync`/`cudaFreeAsync` with a memory pool.

A UMA copy experiment compares `cudaMemcpy`, `cudaMemcpyAsync`, and `cudaMemPrefetchAsync`.

## Build & run

```bash
cd projects/07_streams_events
make all      # build streams_test
make run      # run and tee results to ../../results/07_streams_events.txt
make ptx      # generate PTX
make profile  # Nsight Systems profile
make clean
```

## Key findings

- Pinned non-blocking streams are the fastest for overlapping H2D/compute/D2H.
- `cudaMemPrefetchAsync` on UMA is far faster than explicit `cudaMemcpy` because it
  is a page-table / cache operation rather than a byte-for-byte copy.
- Stream-ordered allocation trades some throughput for memory reuse via the pool.
- A correct sanity check computes the expected output from `h_in[0]` using the
  same recurrence the device applies.

## Artifacts

- `streams_test` — executable (ignored by git)
- `streams.ptx` — generated PTX
- `ANALYSIS.md` — detailed analysis
