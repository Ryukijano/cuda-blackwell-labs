# Project 07: Streams, Events, Async Allocation — Analysis

## Results

| Implementation | Total (ms) | Per Batch (ms) | Peak Mem (MB) | Memory Reuse | Speedup |
|----------------|------------|----------------|---------------|--------------|---------|
| 1. Default stream | 428.7 | 26.8 | 12.0 | No | 1.0x |
| 2. Multiple streams | 3.4 | 0.21 | 48.0 | No | 126x |
| 3. Pinned + non-blocking | 2.8 | 0.18 | 112.0 | No | 152x |
| 4. Stream-ordered alloc | 9.9 | 0.62 | ~48.0 (pool) | Yes | 43x |

## UMA Copy Experiment (64MB)

| Transfer Method | Time (ms) | BW (GB/s) | Notes |
|-----------------|-----------|-----------|-------|
| `cudaMemcpy` H2D | 1.13 | 59.1 | Pageable → pinned staging → device |
| `cudaMemcpyAsync` H2D | 1.13 | 59.6 | Pinned memory, truly asynchronous |
| `cudaMemPrefetchAsync` | 0.068 | 981.8 | ~17x faster — page table / cache flush op |

(Note: earlier runs reported an artificially high prefetch BW because the timer did not wait on the asynchronous prefetch stream; that has been fixed.)

## Key Findings

1. **Default stream is ~150x slower** than pinned multi-stream for batched work. The CPU is
   blocked on every `cudaMemcpy` and every kernel in the default stream.

2. **Multiple streams give a massive speedup** by allowing the GPU to execute copies
   and kernels from different batches concurrently. Even with pageable host memory,
   overlapping launches helps.

3. **Pinned memory is the fastest** (2.8 ms) because `cudaMemcpyAsync` with pinned
   host memory is truly asynchronous and allows full overlap.

4. **Stream-ordered allocation is fastest at memory reuse** but adds ~3.5x overhead vs
   pinned (9.9 ms vs 2.8 ms). The `cudaMallocAsync`/`cudaFreeAsync` per batch is
   not free. Best for when you want to limit peak memory (pool reuse).

5. **On UMA, `cudaMemPrefetchAsync` is ~17x faster** than `cudaMemcpy` because it
   doesn't move bytes — it just changes page ownership/visibility. This is the
   preferred way to migrate data on Grace-Blackwell.

6. **`cudaMemcpy` on UMA is still a real copy** for pageable/pinned memory, with
   bandwidth of ~59 GB/s. Only `cudaMemPrefetchAsync` is a near-instant page table op.

## Analysis of Each Implementation

### Default Stream (428.7 ms)
- `cudaMemcpy` is synchronous: host blocks until complete
- Kernel and copy in same stream: no overlap possible
- Pageable host memory causes driver to allocate pinned staging buffers
- Total CPU blocked time is the entire 428.7 ms
- **Bottleneck: CPU launch + pageable copy overhead**

### Multiple Streams (3.4 ms)
- 4 streams process 4 batches concurrently
- CPU launches all operations quickly, then waits at `cudaDeviceSynchronize()`
- GPU overlaps H2D, compute, D2H from different batches
- Pageable memory still causes some sync fallback, but enough overlap to hide it
- **Bottleneck: Some residual pageable copy overhead**

### Pinned + Non-blocking Streams (2.8 ms)
- `cudaHostAlloc` provides page-locked memory
- `cudaStreamNonBlocking` allows async copies to truly not block
- Best overlap of all four implementations
- Higher peak memory because pinned host memory uses physical pages
- **Bottleneck: None significant — near-optimal**

### Stream-Ordered Allocation (9.9 ms)
- `cudaMallocAsync`/`cudaFreeAsync` per batch adds overhead
- Memory pool reuses freed allocations, so peak device memory is limited
- Good for memory-constrained workloads
- **Bottleneck: Allocation overhead per batch**

## UMA Implications

On GB10 (Grace-Blackwell with unified LPDDR5X):
- `cudaMemcpy` and `cudaMemcpyAsync` **still do real data movement** when using
  pageable or pinned host buffers
- `cudaMemPrefetchAsync` is **the right tool** for data migration
- Managed memory lets you avoid explicit copies, but page faults on first access
  can be slow. Prefetch to avoid faults.

## Recommendations

1. **Never use default stream + `cudaMemcpy` for batched work.**
2. **Use pinned host memory** for H2D/D2H transfers you want to overlap.
3. **Use multiple non-blocking streams** for independent batches.
4. **Use `cudaMemPrefetchAsync`** for managed memory on UMA instead of `cudaMemcpy`.
5. **Use `cudaMallocAsync`/`cudaFreeAsync`** when you want pool-based memory reuse
   and can accept some allocation overhead.
