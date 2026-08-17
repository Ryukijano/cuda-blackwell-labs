#!/usr/bin/env python3
"""
Project 10: Custom Endosight CUDA Kernel
Phase 4 — Application Capstone

PyTorch C++/CUDA extension for point-cloud RGB validity filtering.
Compares:
  1. PyTorch baseline (boolean indexing)
  2. PyTorch vectorized (nonzero + index_select)
  3. Custom CUDA extension with CUB stream compaction

Build:
    conda run -n 3d_recon python setup.py build_ext --inplace

Run:
    conda run -n 3d_recon python benchmark.py
"""

import time
import torch
import numpy as np
import endosight_kernel


def filter_rgb_baseline(points, colors, threshold=0.1):
    """PyTorch baseline with boolean indexing."""
    mask = (colors.sum(dim=1) > threshold)
    return points[mask], colors[mask]


def filter_rgb_vectorized(points, colors, threshold=0.1):
    """PyTorch vectorized with nonzero + index_select."""
    mask = (colors.sum(dim=1) > threshold)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    return points.index_select(0, idx), colors.index_select(0, idx)


def filter_rgb_cuda(points, colors, threshold=0.1):
    """Custom CUDA extension with CUB stream compaction."""
    return endosight_kernel.filter_rgb_valid(points, colors, threshold)


def benchmark(name, fn, points, colors, threshold, repeats=20):
    # warmup
    for _ in range(5):
        _ = fn(points, colors, threshold)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_p, out_c = fn(points, colors, threshold)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "name": name,
        "mean_ms": float(np.mean(times) * 1000),
        "median_ms": float(np.median(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "selected": out_p.size(0),
    }


def main():
    print("=" * 70)
    print("  Project 10: Custom Endosight CUDA Kernel")
    print("  Point-cloud RGB validity filter")
    print("=" * 70)

    sizes = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    threshold = 0.1
    results = []

    for n in sizes:
        print(f"\n  Input size: {n:,} points")

        # Synthetic point cloud
        points = torch.randn(n, 3, device="cuda", dtype=torch.float32)
        colors = torch.rand(n, 3, device="cuda", dtype=torch.float32)

        r1 = benchmark("PyTorch baseline", filter_rgb_baseline, points, colors, threshold)
        r2 = benchmark("PyTorch vectorized", filter_rgb_vectorized, points, colors, threshold)
        r3 = benchmark("CUDA extension", filter_rgb_cuda, points, colors, threshold)

        # Correctness check
        p1, c1 = filter_rgb_baseline(points, colors, threshold)
        p3, c3 = filter_rgb_cuda(points, colors, threshold)
        assert torch.allclose(p1, p3, atol=1e-5), "Output points mismatch"
        assert torch.allclose(c1, c3, atol=1e-5), "Output colors mismatch"

        results.append((n, r1, r2, r3))

    print("\n" + "=" * 80)
    print("  Results Summary")
    print("=" * 80)
    print(f"  {'N':>12} {'Baseline(ms)':>14} {'Vectorized(ms)':>16} {'CUDA(ms)':>12} {'Speedup':>10}")
    print("-" * 80)
    for n, r1, r2, r3 in results:
        speedup = r1["mean_ms"] / r3["mean_ms"]
        print(f"  {n:>12,} {r1['mean_ms']:>14.3f} {r2['mean_ms']:>16.3f} {r3['mean_ms']:>12.3f} {speedup:>10.2f}x")

    print("\n  Notes:")
    print("    - Baseline uses boolean indexing (creates intermediate mask + copies).")
    print("    - Vectorized uses torch.nonzero + index_select (faster baseline).")
    print("    - CUDA extension uses CUB DeviceSelect::Flagged for stream compaction.")
    print("    - On GB10, speedup depends on memory bandwidth and selection ratio.")


if __name__ == "__main__":
    main()
