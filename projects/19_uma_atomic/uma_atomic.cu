/*
 * Project 19: UMA Atomic Coherence Probe
 *
 * Measures cycle-accurate latency of GPU-scope vs system-scope atomics on
 * unified managed memory.  The delta between `atom.global.gpu` and
 * `atom.global.sys` is the NVLink-C2C / hardware-coherence cost on GB10.
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

__global__ void uma_atomic_gpu_kernel(uint32_t* __restrict__ data,
                                      uint64_t* __restrict__ latency,
                                      uint64_t n)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t* ptr = data + tid;
    uint32_t scratch;
    uint64_t t0 = clock64();
    asm volatile("atom.global.gpu.add.u32 %0, [%1], 1;"
                 : "=r"(scratch) : "l"(ptr) : "memory");
    uint64_t t1 = clock64();
    if (scratch == 0xDEADBEEF) latency[tid] = 0;
    else latency[tid] = t1 - t0;
}

__global__ void uma_atomic_sys_kernel(uint32_t* __restrict__ data,
                                      uint64_t* __restrict__ latency,
                                      uint64_t n)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t* ptr = data + tid;
    uint32_t scratch;
    uint64_t t0 = clock64();
    asm volatile("atom.global.sys.add.u32 %0, [%1], 1;"
                 : "=r"(scratch) : "l"(ptr) : "memory");
    uint64_t t1 = clock64();
    if (scratch == 0xDEADBEEF) latency[tid] = 0;
    else latency[tid] = t1 - t0;
}

static void run_pass(uint32_t* data, uint64_t* lat, size_t n,
                     bool sys_scope, int clock_mhz,
                     double& p50_ns, double& p90_ns)
{
    CUDA_CHECK(cudaMemset(data, 0, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(lat,  0, n * sizeof(uint64_t)));

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    if (sys_scope)
        uma_atomic_sys_kernel<<<blocks, threads>>>(data, lat, n);
    else
        uma_atomic_gpu_kernel<<<blocks, threads>>>(data, lat, n);
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
    printf("  UMA Atomic Coherence Probe -- GB10 DGX Spark\n");
    printf("============================================================\n");
    printf("  GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Clock: %d MHz\n\n", clock_mhz);

    size_t n = 64 * 1024;
    uint32_t* data = nullptr;
    uint64_t* lat  = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&lat,  n * sizeof(uint64_t)));

    // Force page allocation with a non-zero pattern before the test.
    CUDA_CHECK(cudaMemset(data, 0xff, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(lat,  0x00, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Prefetch to GPU
    #if CUDART_VERSION >= 12020
    cudaMemLocation loc = {cudaMemLocationTypeDevice, device};
    CUDA_CHECK(cudaMemPrefetchAsync(data, n * sizeof(uint32_t), loc, 0));
    #else
    CUDA_CHECK(cudaMemPrefetchAsync(data, n * sizeof(uint32_t), device, 0));
    #endif
    CUDA_CHECK(cudaDeviceSynchronize());

    const int warmup = 2;
    const int runs = 5;
    double gpu50 = 0, gpu90 = 0, sys50 = 0, sys90 = 0;

    for (int i = 0; i < warmup; ++i)
        run_pass(data, lat, n, false, clock_mhz, gpu50, gpu90);
    for (int i = 0; i < runs; ++i)
        run_pass(data, lat, n, false, clock_mhz, gpu50, gpu90);

    for (int i = 0; i < warmup; ++i)
        run_pass(data, lat, n, true, clock_mhz, sys50, sys90);
    for (int i = 0; i < runs; ++i)
        run_pass(data, lat, n, true, clock_mhz, sys50, sys90);

    printf("  atom.global.gpu  p50: %8.2f ns  p90: %8.2f ns\n", gpu50, gpu90);
    printf("  atom.global.sys  p50: %8.2f ns  p90: %8.2f ns\n", sys50, sys90);
    double ratio = (gpu50 > 0) ? sys50 / gpu50 : 0.0;
    printf("  SYS/GPU ratio    : %.2fx\n", ratio);
    printf("  Coherence overhead: %.2f ns\n", sys50 - gpu50);
    printf("\n");

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(lat));
    return 0;
}
