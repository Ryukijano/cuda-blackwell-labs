// CUDA Blackwell Labs — Benchmark Utilities
// Helper functions for running repeated benchmarks with warmup, statistics,
// and bandwidth/compute throughput calculations.

#pragma once

#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// ============================================================================
// Benchmark Runner
// ============================================================================

struct BenchmarkResult {
    std::string name;
    int iterations;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    double median_ms;

    // Derived metrics (set by caller)
    double bandwidth_gbps;   // effective bandwidth (GB/s)
    double throughput_gflops; // compute throughput (GFLOP/s)
    size_t bytes_moved;      // total bytes (read + write)
    double flops;            // total FLOPs

    void print() const {
        printf("  %-25s  mean=%8.3f ms  std=%7.3f  min=%8.3f  max=%8.3f  median=%8.3f\n",
               name.c_str(), mean_ms, std_ms, min_ms, max_ms, median_ms);
        if (bandwidth_gbps > 0) {
            printf("    bandwidth:    %.1f GB/s  (%.1f%% of peak %.0f GB/s)\n",
                   bandwidth_gbps, 100.0 * bandwidth_gbps / GB10_PEAK_BW_GBPS, GB10_PEAK_BW_GBPS);
        }
        if (throughput_gflops > 0) {
            printf("    throughput:   %.1f GFLOP/s\n", throughput_gflops);
        }
    }
};

// Run a lambda function `iterations` times with warmup, return statistics
template<typename Func>
BenchmarkResult benchmark(const std::string& name, int warmup, int iterations, Func&& func) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        func();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    std::vector<double> times(iterations);
    GpuTimer timer;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        func();
        timer.stop();
        times[i] = timer.milliseconds();
    }

    // Statistics
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / iterations;
    double sq_sum = std::accumulate(times.begin(), times.end(), 0.0,
        [mean](double acc, double v) { return acc + (v - mean) * (v - mean); });
    double std_dev = std::sqrt(sq_sum / iterations);

    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];

    return BenchmarkResult{
        .name = name,
        .iterations = iterations,
        .mean_ms = mean,
        .std_ms = std_dev,
        .min_ms = sorted.front(),
        .max_ms = sorted.back(),
        .median_ms = median,
        .bandwidth_gbps = 0,
        .throughput_gflops = 0,
        .bytes_moved = 0,
        .flops = 0,
    };
}

// ============================================================================
// Array Initialization Helpers
// ============================================================================

template<typename T>
void init_random(T* arr, size_t n, T min_val = 0, T max_val = 1) {
    for (size_t i = 0; i < n; i++) {
        if constexpr (std::is_floating_point_v<T>) {
            arr[i] = min_val + (T)rand() / RAND_MAX * (max_val - min_val);
        } else {
            arr[i] = (T)(min_val + rand() % ((int)max_val - (int)min_val + 1));
        }
    }
}

template<typename T>
bool verify_equal(const T* a, const T* b, size_t n, T tolerance = 1e-4) {
    for (size_t i = 0; i < n; i++) {
        if constexpr (std::is_floating_point_v<T>) {
            if (fabs((double)a[i] - (double)b[i]) > tolerance) {
                printf("  MISMATCH at index %zu: %f vs %f (diff=%f)\n",
                       i, (double)a[i], (double)b[i], fabs((double)a[i] - (double)b[i]));
                return false;
            }
        } else {
            if (a[i] != b[i]) {
                printf("  MISMATCH at index %zu: %d vs %d\n", (int)i, (int)a[i], (int)b[i]);
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Comparison Table Printer
// ============================================================================

class ResultsTable {
public:
    void add(const BenchmarkResult& r) { results_.push_back(r); }

    void print() const {
        printf("\n  %-25s  %12s  %12s  %12s  %12s\n",
               "Kernel", "Mean (ms)", "Median (ms)", "BW (GB/s)", "% Peak");
        printf("  %-25s  %12s  %12s  %12s  %12s\n",
               "-------------------------", "------------", "------------", "------------", "------------");
        for (auto& r : results_) {
            double pct = r.bandwidth_gbps > 0 ? 100.0 * r.bandwidth_gbps / GB10_PEAK_BW_GBPS : 0;
            printf("  %-25s  %12.3f  %12.3f  %12.1f  %11.1f%%\n",
                   r.name.c_str(), r.mean_ms, r.median_ms, r.bandwidth_gbps, pct);
        }
    }

    void print_with_flops() const {
        printf("\n  %-25s  %12s  %12s  %12s  %12s\n",
               "Kernel", "Mean (ms)", "BW (GB/s)", "GFLOP/s", "% Peak BW");
        printf("  %-25s  %12s  %12s  %12s  %12s\n",
               "-------------------------", "------------", "------------", "------------", "------------");
        for (auto& r : results_) {
            double pct = r.bandwidth_gbps > 0 ? 100.0 * r.bandwidth_gbps / GB10_PEAK_BW_GBPS : 0;
            printf("  %-25s  %12.3f  %12.1f  %12.1f  %11.1f%%\n",
                   r.name.c_str(), r.mean_ms, r.bandwidth_gbps, r.throughput_gflops, pct);
        }
    }

private:
    std::vector<BenchmarkResult> results_;
};
