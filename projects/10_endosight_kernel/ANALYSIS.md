# Project 10: Custom Endosight CUDA Kernel — Analysis

## Results: Point-Cloud RGB Validity Filter

| N | PyTorch Baseline (ms) | PyTorch Vectorized (ms) | CUDA Extension (ms) | Speedup vs baseline |
|---|------------------------|--------------------------|----------------------|---------------------|
| 100,000 | 0.070 | 0.050 | 0.243 | 0.29x |
| 500,000 | 0.267 | 0.202 | 0.373 | 0.72x |
| 1,000,000 | 0.500 | 0.431 | 0.583 | 0.86x |
| 2,000,000 | 1.060 | 0.887 | 0.841 | 1.26x |
| 5,000,000 | 2.676 | 2.269 | 2.052 | **1.30x** |

## What Was Built

A PyTorch C++/CUDA extension that performs **stream compaction** on a point cloud:
- **Input**: `points` (N×3 float) and `colors` (N×3 float) on GPU
- **Filter**: keep points where `r + g + b > threshold`
- **Output**: compacted `out_points` (M×3) and `out_colors` (M×3)

## Implementation Details

1. **Custom CUDA kernel** computes the validity flag for each point in parallel.
2. **CUB `DeviceSelect::Flagged`** performs global stream compaction on indices.
3. **Custom gather kernel** fetches `points[idx]` and `colors[idx]` in one pass.
4. **Wrapped** with `torch.utils.cpp_extension.CUDAExtension` and bound via pybind11.

## Why It Only Wins at Large N

- For small N, the custom kernel launch + CUB setup overhead dominates.
- For N ≥ 2M, the fused CUDA path amortizes overhead and uses less host synchronization.
- PyTorch's `torch.index_select` is already highly optimized; matching it is hard.
- The real benefit of a custom CUDA kernel is **control over fusion and memory layout**,
  not a 10x speedup on this simple operation.

## Lesson for Endosight

- A custom CUDA extension is worth it when:
  - The operation runs **per frame** or **per point cloud** (millions of calls).
  - PyTorch cannot **fuse** multiple steps (e.g., flag + compact + gather).
  - You need **non-standard layouts** or **warp-level operations**.
- For one-off memory copies or simple indexing, PyTorch's built-ins are already optimal.

## Recommendations

1. **Use CUB** for stream compaction, not hand-rolled prefix sums, unless you have
   a very specific access pattern.
2. **Keep data in float3/float4** layouts to maximize coalescing when writing custom
   kernels.
3. **Profile with Nsight Compute** to verify that memory bandwidth is saturated.
4. **Only write a custom CUDA extension when the operation is repeated millions of
   times** or the speedup is ≥1.5x in isolation.
