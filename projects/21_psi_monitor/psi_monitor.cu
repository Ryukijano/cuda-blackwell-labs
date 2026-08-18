/*
 * Project 21: PSI / Memory Stall Monitor
 *
 * Samples /proc/pressure/memory while a GPU kernel hammers unified memory.
 * PSI is often the only reliable stall signal on GB10 because NVML and
 * Nsight UVM tracing are not fully supported on this hardware.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include "cuda_utils.h"

__global__ void gpu_hammer_kernel(const float* __restrict__ buf,
                                  float* __restrict__ sink,
                                  size_t n, int iters)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (int r = 0; r < iters; ++r) {
        for (size_t i = idx; i < n; i += stride) {
            float val;
            asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(buf + i));
            acc += val;
        }
    }
    sink[idx] = acc;
}

static uint64_t parse_psi_total(const char* line)
{
    const char* p = strstr(line, "total=");
    if (!p) return 0;
    return strtoull(p + 6, nullptr, 10);
}

static bool read_psi(uint64_t& some_total, uint64_t& full_total)
{
    FILE* f = fopen("/proc/pressure/memory", "r");
    if (!f) return false;
    char line[256];
    some_total = full_total = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "some", 4) == 0) some_total = parse_psi_total(line);
        if (strncmp(line, "full", 4) == 0) full_total = parse_psi_total(line);
    }
    fclose(f);
    return true;
}

struct MonitorArg {
    std::atomic<bool> stop;
    uint64_t some_us;
    uint64_t full_us;
    uint64_t start_us;
    uint64_t end_us;
    double max_some_pct;
    double max_full_pct;
};

static void monitor_fn(MonitorArg* arg)
{
    uint64_t start_some, start_full;
    if (!read_psi(start_some, start_full)) { arg->stop = true; return; }
    arg->some_us = start_some;
    arg->full_us = start_full;

    auto t0 = std::chrono::steady_clock::now();
    arg->start_us = 0;
    arg->max_some_pct = arg->max_full_pct = 0.0;

    while (!arg->stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        uint64_t cur_some, cur_full;
        if (!read_psi(cur_some, cur_full)) continue;
        auto now = std::chrono::steady_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(now - t0).count();
        double some_pct = (double)(cur_some - start_some) / elapsed_us * 100.0;
        double full_pct = (double)(cur_full - start_full) / elapsed_us * 100.0;
        if (some_pct > arg->max_some_pct) arg->max_some_pct = some_pct;
        if (full_pct > arg->max_full_pct) arg->max_full_pct = full_pct;
    }

    uint64_t final_some, final_full;
    if (read_psi(final_some, final_full)) {
        arg->some_us = final_some - start_some;
        arg->full_us = final_full - start_full;
    }
    auto t1 = std::chrono::steady_clock::now();
    arg->end_us = (uint64_t)std::chrono::duration<double, std::micro>(t1 - t0).count();
}

int main(int argc, char** argv)
{
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("============================================================\n");
    printf("  PSI / Memory Stall Monitor -- GB10 DGX Spark\n");
    printf("============================================================\n");
    printf("  GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    size_t bytes = (argc >= 2) ? (size_t)atoi(argv[1]) * 1024 * 1024
                               : (size_t)256 * 1024 * 1024;
    size_t n = bytes / sizeof(float);
    int iters = (argc >= 3) ? atoi(argv[2]) : 50;

    float* buf = nullptr;
    CUDA_CHECK(cudaMallocManaged(&buf, bytes));
    // Non-zero pattern to avoid read-only zero-page fast-paths in UVM/HMM.
    CUDA_CHECK(cudaMemset(buf, 0x3f, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    #if CUDART_VERSION >= 12020
    cudaMemLocation loc = {cudaMemLocationTypeDevice, device};
    CUDA_CHECK(cudaMemPrefetchAsync(buf, bytes, loc, 0));
    #else
    CUDA_CHECK(cudaMemPrefetchAsync(buf, bytes, device, 0));
    #endif
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 2048) blocks = 2048;

    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, (size_t)blocks * threads * sizeof(float)));

    MonitorArg mon = {};
    mon.stop = false;
    std::thread mon_thread(monitor_fn, &mon);

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    gpu_hammer_kernel<<<blocks, threads>>>(buf, sink, n, iters);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));

    mon.stop = true;
    mon_thread.join();

    double total_gb = (double)(n * sizeof(float) * iters) / (1024.0 * 1024.0 * 1024.0);
    double time_s = ms / 1000.0;
    double bw_gbs = total_gb / time_s;

    printf("  Buffer: %.1f MB\n", bytes / (1024.0 * 1024.0));
    printf("  Iterations: %d\n", iters);
    printf("  GPU time: %.3f s\n", time_s);
    printf("  GPU read BW: %.2f GB/s\n", bw_gbs);
    printf("  PSI some total: %.3f ms (%.4f%% of runtime)\n",
           mon.some_us / 1000.0, (double)mon.some_us / (mon.end_us * 10.0));
    printf("  PSI full total: %.3f ms (%.4f%% of runtime)\n",
           mon.full_us / 1000.0, (double)mon.full_us / (mon.end_us * 10.0));
    printf("  PSI some peak (decayed): %.4f%%\n", mon.max_some_pct);
    printf("  PSI full peak (decayed): %.4f%%\n", mon.max_full_pct);
    printf("\n");

    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(buf));
    return 0;
}
