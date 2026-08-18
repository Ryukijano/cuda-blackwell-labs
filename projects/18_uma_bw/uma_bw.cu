/*
 * Project 18: UMA PTX Bandwidth Probe
 *
 * Measures true LPDDR5X bandwidth on GB10 using PTX cache operators:
 *   ld.global.cg  -- cache at L2, bypass L1 (read)
 *   st.global.cs  -- bypass L2, streaming store (write)
 *
 * On GB10 the GPU and CPU share the same physical LPDDR5X, so we also
 * optionally report CPU read bandwidth after prefetching to the host.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "cuda_utils.h"

__global__ void gpu_read_kernel(const float* __restrict__ buf,
                                float* __restrict__ sink, size_t n)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        float val;
        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(buf + i));
        acc += val;
    }
    sink[idx] = acc;
}

__global__ void gpu_write_kernel(float* __restrict__ buf,
                                 size_t n, float v)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride)
        asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(buf + i), "f"(v));
}

__global__ void gpu_copy_kernel(float* __restrict__ dst,
                                const float* __restrict__ src,
                                size_t n)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        float val;
        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(src + i));
        asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(dst + i), "f"(val));
    }
}

static double timed_run_read(float* a, float* sink,
                             size_t n, int blocks, int threads)
{
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    gpu_read_kernel<<<blocks, threads>>>(a, sink, n);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return (double)(n * sizeof(float)) / (ms * 1e6); // GB/s
}

static double timed_run_write(float* a, size_t n, int blocks, int threads)
{
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    gpu_write_kernel<<<blocks, threads>>>(a, n, 1.0f);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return (double)(n * sizeof(float)) / (ms * 1e6); // GB/s
}

static double timed_run_copy(float* dst, const float* src,
                             size_t n, int blocks, int threads)
{
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    gpu_copy_kernel<<<blocks, threads>>>(dst, src, n);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return (double)(n * sizeof(float) * 2) / (ms * 1e6); // GB/s
}

int main(int argc, char** argv)
{
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("============================================================\n");
    printf("  UMA PTX Bandwidth Probe -- GB10 DGX Spark\n");
    printf("============================================================\n");
    printf("  GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    size_t bytes = (argc >= 2) ? (size_t)atoi(argv[1]) * 1024 * 1024
                               : (size_t)256 * 1024 * 1024;
    size_t n = bytes / sizeof(float);

    float *a = nullptr, *b = nullptr;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    // Fill with a non-zero byte pattern so the pages are truly allocated and
    // not satisfied by a read-only shared zero page (which makes reads look
    // instant on some UVM/HMM paths).
    CUDA_CHECK(cudaMemset(a, 0x3f, bytes));
    CUDA_CHECK(cudaMemset(b, 0x3f, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Prefetch to GPU
    #if CUDART_VERSION >= 12020
    cudaMemLocation loc = {cudaMemLocationTypeDevice, device};
    CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, loc, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, loc, 0));
    #else
    CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, device, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, device, 0));
    #endif
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 2048) blocks = 2048;

    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, (size_t)blocks * threads * sizeof(float)));

    const int warmup = 2;
    const int runs = 5;

    double rsum = 0, wsum = 0, csum = 0;
    for (int i = 0; i < warmup; ++i) timed_run_read(a, sink, n, blocks, threads);
    for (int i = 0; i < runs; ++i) rsum += timed_run_read(a, sink, n, blocks, threads);
    for (int i = 0; i < warmup; ++i) timed_run_write(a, n, blocks, threads);
    for (int i = 0; i < runs; ++i) wsum += timed_run_write(a, n, blocks, threads);
    for (int i = 0; i < warmup; ++i) timed_run_copy(b, a, n, blocks, threads);
    for (int i = 0; i < runs; ++i) csum += timed_run_copy(b, a, n, blocks, threads);

    printf("\n  Buffer: %.1f MB (%zu floats)\n", bytes / (1024.0 * 1024.0), n);
    printf("  Blocks: %d  Threads: %d\n\n", blocks, threads);
    printf("  GPU read  (ld.global.cg): %.2f GB/s\n", rsum / runs);
    printf("  GPU write (st.global.cs): %.2f GB/s\n", wsum / runs);
    printf("  GPU copy  (cg + cs)     : %.2f GB/s\n", csum / runs);
    printf("\n");

    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    return 0;
}
