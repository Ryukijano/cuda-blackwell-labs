/*
 * Project 20: UMA Fault / Residency Latency Probe
 *
 * Measures UMA page-fault and residency latency with `ld.global.cv.f32`
 * and `clock64()`.  Three passes:
 *   COLD    -- CPU touches all pages, then GPU accesses them (fault path)
 *   WARM    -- pages are prefetched to the GPU
 *   PRESSURE-- half the array is CPU-resident, half is GPU-resident
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "cuda_utils.h"

static double percentile(const std::vector<uint64_t>& v, double pct)
{
    if (v.empty()) return 0;
    size_t idx = (size_t)(pct / 100.0 * (v.size() - 1));
    if (idx >= v.size()) idx = v.size() - 1;
    return (double)v[idx];
}

__global__ void uma_fault_probe_kernel(const float* __restrict__ data,
                                       uint64_t* __restrict__ latency,
                                       uint64_t n)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const float* ptr = data + tid;
    float val;
    uint64_t t0 = clock64();
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(val) : "l"(ptr) : "memory");
    uint64_t t1 = clock64();
    if (val != val) latency[tid] = 0; // never true
    else latency[tid] = t1 - t0;
}

enum PassType { PASS_COLD, PASS_WARM, PASS_PRESSURE };

static void run_pass(float* data, uint64_t* lat, size_t n,
                     PassType pass, int clock_mhz, int device,
                     double& p50_ns, double& p90_ns)
{
    if (pass == PASS_COLD) {
        for (size_t i = 0; i < n; ++i) data[i] = (float)i;
    } else if (pass == PASS_WARM) {
        #if CUDART_VERSION >= 12020
        cudaMemLocation loc = {cudaMemLocationTypeDevice, device};
        CUDA_CHECK(cudaMemPrefetchAsync(data, n * sizeof(float), loc, 0));
        #else
        CUDA_CHECK(cudaMemPrefetchAsync(data, n * sizeof(float), device, 0));
        #endif
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        for (size_t i = 0; i < n / 2; ++i) data[i] = (float)i;
        #if CUDART_VERSION >= 12020
        cudaMemLocation loc = {cudaMemLocationTypeDevice, device};
        CUDA_CHECK(cudaMemPrefetchAsync(data + n / 2, (n / 2) * sizeof(float), loc, 0));
        #else
        CUDA_CHECK(cudaMemPrefetchAsync(data + n / 2, (n / 2) * sizeof(float), device, 0));
        #endif
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemset(lat, 0, n * sizeof(uint64_t)));
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    uma_fault_probe_kernel<<<blocks, threads>>>(data, lat, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> host(n);
    std::vector<uint64_t> valid;
    CUDA_CHECK(cudaMemcpy(host.data(), lat, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (auto v : host) if (v > 0) valid.push_back(v);
    std::sort(valid.begin(), valid.end());

    double p50_cyc = percentile(valid, 50.0);
    double p90_cyc = percentile(valid, 90.0);
    p50_ns = p50_cyc / clock_mhz * 1000.0;
    p90_ns = p90_cyc / clock_mhz * 1000.0;
}

int main(int argc, char** argv)
{
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
    int clock_mhz = clock_khz / 1000;

    printf("============================================================\n");
    printf("  UMA Fault / Residency Latency Probe -- GB10 DGX Spark\n");
    printf("============================================================\n");
    printf("  GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Clock: %d MHz\n\n", clock_mhz);

    size_t n = 64 * 1024 * 1024 / sizeof(float);
    float* data = nullptr;
    uint64_t* lat = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&lat,  n * sizeof(uint64_t)));

    double c50, c90, w50, w90, p50, p90;
    run_pass(data, lat, n, PASS_COLD, clock_mhz, device, c50, c90);
    run_pass(data, lat, n, PASS_WARM, clock_mhz, device, w50, w90);
    run_pass(data, lat, n, PASS_PRESSURE, clock_mhz, device, p50, p90);

    printf("  COLD    (CPU touches)  p50: %8.2f ns  p90: %8.2f ns\n", c50, c90);
    printf("  WARM    (GPU resident) p50: %8.2f ns  p90: %8.2f ns\n", w50, w90);
    printf("  PRESSURE (mixed)       p50: %8.2f ns  p90: %8.2f ns\n", p50, p90);
    double ratio = (w50 > 0) ? c50 / w50 : 0.0;
    printf("  COLD/WARM ratio: %.2fx\n", ratio);
    printf("\n");

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(lat));
    return 0;
}
