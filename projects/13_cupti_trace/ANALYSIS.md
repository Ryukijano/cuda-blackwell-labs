# Project 13: CUPTI Activity Trace — Analysis

## Results

| Kernel | Duration | Grid | Block |
|--------|----------|------|-------|
| `warmup_kernel` | 14.75 µs (cold) | 4096,1,1 | 256,1,1 |
| `warmup_kernel` | ~5 µs (warm) | 4096,1,1 | 256,1,1 |

- CUPTI correctly records the kernel name (mangled), launch dimensions, and wall-clock duration.
- The first launch is ~3× slower due to cold-cache and launch bookkeeping.

## Takeaways

1. CUPTI activity tracing works on GB10 without profiling counter permissions.
2. The activity trace is asynchronous and uses CUPTI's worker thread; buffers are delivered through the `bufferCompleted` callback.
3. `cuptiActivityFlushAll(1)` blocks until all pending records are delivered; it should not be called from inside the completion callback.

## Counter-sampler note

The CUPTI Range Profiler (counters/metrics) was attempted but is blocked by `ERR_NVGPUCTRPERM / CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` on this system. To enable it, the process needs `CAP_SYS_ADMIN` or the system administrator must change the permissions via `nvidia-modprobe`/`/proc` knobs. An activity trace is the practical, non-privileged alternative.
