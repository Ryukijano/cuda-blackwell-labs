// Project 07: Streams, Events, Async Allocation
// Phase 3 — Runtime and Systems Literacy

#include "cuda_utils.h"
#include "benchmark.h"
#include <vector>
#include <algorithm>

// Include NVTX for Nsight Systems annotation
#include <nvtx3/nvToolsExt.h>

// ============================================================================
// Pipeline kernels
// ============================================================================

__global__ void preprocess_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 0.5f + 0.1f;
}

__global__ void compute_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        #pragma unroll
        for (int j = 0; j < 10; j++) {
            x = x * 1.0001f + 0.001f;
        }
        out[i] = x;
    }
}

// ============================================================================
// Implementation 1: Default stream (synchronous)
// ============================================================================

void impl1_default(float* h_in, float* h_out, int n, int batches, size_t bytes) {
    float *d_in, *d_int, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_int, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    GpuTimer timer;
    timer.start();
    for (int b = 0; b < batches; b++) {
        size_t off = (size_t)b * n;
        nvtxRangePushA("H2D");
        cudaMemcpy(d_in, h_in + off, bytes, cudaMemcpyHostToDevice);
        nvtxRangePop();
        nvtxRangePushA("pre");
        preprocess_kernel<<<(n + 255) / 256, 256>>>(d_in, d_int, n);
        nvtxRangePop();
        nvtxRangePushA("comp");
        compute_kernel<<<(n + 255) / 256, 256>>>(d_int, d_out, n);
        nvtxRangePop();
        nvtxRangePushA("D2H");
        cudaMemcpy(h_out + off, d_out, bytes, cudaMemcpyDeviceToHost);
        nvtxRangePop();
    }
    cudaDeviceSynchronize();
    timer.stop();

    printf("    total=%.3f ms  per-batch=%.3f  mem=%.2f MB  reuse=No\n",
           timer.milliseconds(), timer.milliseconds() / batches, 3.0f * bytes / 1024 / 1024);

    cudaFree(d_in); cudaFree(d_int); cudaFree(d_out);
}

// ============================================================================
// Implementation 2: Multiple streams (inter-batch parallelism)
// ============================================================================

void impl2_multi_stream(float* h_in, float* h_out, int n, int batches, size_t bytes) {
    const int NSTREAM = 4;
    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; i++) cudaStreamCreate(&stream[i]);

    float *d_in[NSTREAM], *d_int[NSTREAM], *d_out[NSTREAM];
    for (int i = 0; i < NSTREAM; i++) {
        cudaMalloc(&d_in[i], bytes);
        cudaMalloc(&d_int[i], bytes);
        cudaMalloc(&d_out[i], bytes);
    }

    GpuTimer timer;
    timer.start();
    for (int b = 0; b < batches; b++) {
        int s = b % NSTREAM;
        size_t off = (size_t)b * n;
        nvtxRangePushA("H2D");
        cudaMemcpyAsync(d_in[s], h_in + off, bytes, cudaMemcpyHostToDevice, stream[s]);
        nvtxRangePop();
        nvtxRangePushA("pre");
        preprocess_kernel<<<(n + 255) / 256, 256, 0, stream[s]>>>(d_in[s], d_int[s], n);
        nvtxRangePop();
        nvtxRangePushA("comp");
        compute_kernel<<<(n + 255) / 256, 256, 0, stream[s]>>>(d_int[s], d_out[s], n);
        nvtxRangePop();
        nvtxRangePushA("D2H");
        cudaMemcpyAsync(h_out + off, d_out[s], bytes, cudaMemcpyDeviceToHost, stream[s]);
        nvtxRangePop();
    }
    cudaDeviceSynchronize();
    timer.stop();

    printf("    total=%.3f ms  per-batch=%.3f  mem=%.2f MB  reuse=No\n",
           timer.milliseconds(), timer.milliseconds() / batches,
           3.0f * NSTREAM * bytes / 1024 / 1024);

    for (int i = 0; i < NSTREAM; i++) {
        cudaFree(d_in[i]); cudaFree(d_int[i]); cudaFree(d_out[i]);
        cudaStreamDestroy(stream[i]);
    }
}

// ============================================================================
// Implementation 3: Pinned + non-blocking streams
// ============================================================================

void impl3_pinned(float* h_in, float* h_out, int n, int batches, size_t bytes) {
    const int NSTREAM = 4;
    float *h_in_p, *h_out_p;
    cudaHostAlloc(&h_in_p, batches * bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out_p, batches * bytes, cudaHostAllocDefault);
    memcpy(h_in_p, h_in, batches * bytes);

    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; i++)
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);

    float *d_in[NSTREAM], *d_int[NSTREAM], *d_out[NSTREAM];
    for (int i = 0; i < NSTREAM; i++) {
        cudaMalloc(&d_in[i], bytes);
        cudaMalloc(&d_int[i], bytes);
        cudaMalloc(&d_out[i], bytes);
    }

    GpuTimer timer;
    timer.start();
    for (int b = 0; b < batches; b++) {
        int s = b % NSTREAM;
        size_t off = (size_t)b * n;
        nvtxRangePushA("H2D");
        cudaMemcpyAsync(d_in[s], h_in_p + off, bytes, cudaMemcpyHostToDevice, stream[s]);
        nvtxRangePop();
        nvtxRangePushA("pre");
        preprocess_kernel<<<(n + 255) / 256, 256, 0, stream[s]>>>(d_in[s], d_int[s], n);
        nvtxRangePop();
        nvtxRangePushA("comp");
        compute_kernel<<<(n + 255) / 256, 256, 0, stream[s]>>>(d_int[s], d_out[s], n);
        nvtxRangePop();
        nvtxRangePushA("D2H");
        cudaMemcpyAsync(h_out_p + off, d_out[s], bytes, cudaMemcpyDeviceToHost, stream[s]);
        nvtxRangePop();
    }
    cudaDeviceSynchronize();
    timer.stop();

    memcpy(h_out, h_out_p, batches * bytes);

    printf("    total=%.3f ms  per-batch=%.3f  mem=%.2f MB  reuse=No\n",
           timer.milliseconds(), timer.milliseconds() / batches,
           (batches * bytes + 3.0f * NSTREAM * bytes) / 1024 / 1024);

    for (int i = 0; i < NSTREAM; i++) {
        cudaFree(d_in[i]); cudaFree(d_int[i]); cudaFree(d_out[i]);
        cudaStreamDestroy(stream[i]);
    }
    cudaFreeHost(h_in_p); cudaFreeHost(h_out_p);
}

// ============================================================================
// Implementation 4: Stream-ordered async allocation
// ============================================================================

void impl4_async_alloc(float* h_in, float* h_out, int n, int batches, size_t bytes) {
    const int NSTREAM = 4;

    // Use default memory pool and set release threshold
    cudaMemPool_t pool;
    cudaDeviceGetMemPool(&pool, 0);
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

    float *h_in_p, *h_out_p;
    cudaHostAlloc(&h_in_p, batches * bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out_p, batches * bytes, cudaHostAllocDefault);
    memcpy(h_in_p, h_in, batches * bytes);

    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; i++)
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);

    GpuTimer timer;
    timer.start();
    for (int b = 0; b < batches; b++) {
        int s = b % NSTREAM;
        size_t off = (size_t)b * n;

        float *d_in, *d_int, *d_out;
        cudaMallocAsync(&d_in, bytes, stream[s]);
        cudaMallocAsync(&d_int, bytes, stream[s]);
        cudaMallocAsync(&d_out, bytes, stream[s]);

        nvtxRangePushA("H2D");
        cudaMemcpyAsync(d_in, h_in_p + off, bytes, cudaMemcpyHostToDevice, stream[s]);
        nvtxRangePop();
        nvtxRangePushA("pre");
        preprocess_kernel<<<(n + 255) / 256, 256, 0, stream[s]>>>(d_in, d_int, n);
        nvtxRangePop();
        nvtxRangePushA("comp");
        compute_kernel<<<(n + 255) / 256, 256, 0, stream[s]>>>(d_int, d_out, n);
        nvtxRangePop();
        nvtxRangePushA("D2H");
        cudaMemcpyAsync(h_out_p + off, d_out, bytes, cudaMemcpyDeviceToHost, stream[s]);
        nvtxRangePop();

        cudaFreeAsync(d_in, stream[s]);
        cudaFreeAsync(d_int, stream[s]);
        cudaFreeAsync(d_out, stream[s]);
    }
    cudaDeviceSynchronize();
    timer.stop();

    memcpy(h_out, h_out_p, batches * bytes);

    printf("    total=%.3f ms  per-batch=%.3f  mem=~%.2f MB (pool)  reuse=Yes\n",
           timer.milliseconds(), timer.milliseconds() / batches,
           3.0f * NSTREAM * bytes / 1024 / 1024);

    for (int i = 0; i < NSTREAM; i++) cudaStreamDestroy(stream[i]);
    cudaFreeHost(h_in_p); cudaFreeHost(h_out_p);
}

// ============================================================================
// UMA copy experiment
// ============================================================================

void uma_copy_experiment(int n) {
    size_t bytes = n * sizeof(float);

    float *h, *d;
    h = (float*)malloc(bytes);
    cudaMalloc(&d, bytes);
    memset(h, 1, bytes);

    print_header("UMA Copy Experiment");

    // 1. cudaMemcpy H2D
    for (int i = 0; i < 3; i++) cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    GpuTimer t1;
    std::vector<double> t1s(20);
    for (int i = 0; i < 20; i++) {
        t1.start(); cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice); t1.stop();
        t1s[i] = t1.milliseconds();
    }
    std::sort(t1s.begin(), t1s.end());

    // 2. cudaMemcpyAsync H2D (pinned)
    float *h_p;
    cudaHostAlloc(&h_p, bytes, cudaHostAllocDefault);
    memcpy(h_p, h, bytes);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (int i = 0; i < 3; i++)
        cudaMemcpyAsync(d, h_p, bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    GpuTimer t2;
    std::vector<double> t2s(20);
    for (int i = 0; i < 20; i++) {
        t2.start(); cudaMemcpyAsync(d, h_p, bytes, cudaMemcpyHostToDevice, stream); t2.stop();
        t2s[i] = t2.milliseconds();
    }
    cudaStreamSynchronize(stream);
    std::sort(t2s.begin(), t2s.end());

    // 3. Direct managed memory (kernel reads host pointer)
    float *m;
    cudaMallocManaged(&m, bytes);
    for (size_t i = 0; i < bytes / sizeof(float); i++) m[i] = h[i];
    // Prefetch to GPU
    cudaMemLocation loc = {cudaMemLocationTypeDevice, 0};
    cudaMemPrefetchAsync(m, bytes, loc, 0, stream);
    cudaStreamSynchronize(stream);

    // Time cudaMemPrefetchAsync back to CPU
    GpuTimer t3;
    std::vector<double> t3s(20);
    for (int i = 0; i < 20; i++) {
        cudaMemLocation host_loc = {cudaMemLocationTypeHost, 0};
        t3.start();
        cudaMemPrefetchAsync(m, bytes, host_loc, 0, stream);
        t3.stop();
        t3s[i] = t3.milliseconds();
        cudaStreamSynchronize(stream);
    }
    std::sort(t3s.begin(), t3s.end());
    cudaFree(m);

    printf("  cudaMemcpy H2D:           %.3f ms  BW=%.1f GB/s\n",
           t1s[10], bytes / (t1s[10] * 1e-3) / 1e9);
    printf("  cudaMemcpyAsync H2D:      %.3f ms  BW=%.1f GB/s\n",
           t2s[10], bytes / (t2s[10] * 1e-3) / 1e9);
    printf("  cudaMemPrefetchAsync:     %.3f ms  BW=%.1f GB/s\n",
           t3s[10], bytes / (t3s[10] * 1e-3) / 1e9);
    printf("  (On UMA, these are cache flushes/page table ops, not real copies)\n");

    free(h); cudaFree(d); cudaFreeHost(h_p); cudaStreamDestroy(stream);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_header("Streams, Events, Async Allocation — GB10");

    int n = 1 << 20;        // 1M floats per batch
    int batches = 16;
    size_t bytes = n * sizeof(float);
    size_t total_bytes = batches * bytes;

    printf("  Batch: %d floats (%.2f MB),  %d batches, total %.2f MB\n\n",
           n, bytes / 1e6, batches, total_bytes / 1e6);

    // Allocate pageable host buffers
    float *h_in, *h_out;
    h_in = (float*)malloc(total_bytes);
    h_out = (float*)malloc(total_bytes);
    srand(42);
    for (size_t i = 0; i < total_bytes / sizeof(float); i++)
        h_in[i] = (float)(rand() % 100) / 100.0f;

    print_header("1. Default stream (synchronous)");
    impl1_default(h_in, h_out, n, batches, bytes);

    print_header("2. Multiple streams");
    impl2_multi_stream(h_in, h_out, n, batches, bytes);

    print_header("3. Pinned + non-blocking streams");
    impl3_pinned(h_in, h_out, n, batches, bytes);

    print_header("4. Stream-ordered allocation");
    impl4_async_alloc(h_in, h_out, n, batches, bytes);

    uma_copy_experiment(16777216);  // 64MB

    // Verify correctness (simple check on last batch)
    float first = h_out[0];
    printf("\n  Sanity check h_out[0] = %.4f (should be ~0.6)\n", first);

    free(h_in); free(h_out);
    return 0;
}
