/*
 * bandwidth_test.cu - CUDA Memory Bandwidth & Latency Lab (Project 2)
 *
 * Security design:
 * - No shell execution, no user-supplied format strings.
 * - Sizes validated for overflow before allocation.
 * - All CUDA calls checked.
 */

#include <cuda_runtime_api.h>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <filesystem>
#include <system_error>

static const char* LOG_PATH = "results/bandwidth_log.txt";
static FILE* g_log = nullptr;

static void log_printf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void log_printf(const char* fmt, ...)
{
    va_list a, b;
    va_start(a, fmt);
    va_copy(b, a);
    vprintf(fmt, a);
    if (g_log) vfprintf(g_log, fmt, b);
    va_end(b);
    va_end(a);
}

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            log_printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                       cudaGetErrorString(err));                               \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

enum class AllocType { DEVICE, MANAGED, PINNED, MALLOC_REGISTERED };

static const char* alloc_name(AllocType t)
{
    switch (t) {
        case AllocType::DEVICE: return "cudaMalloc";
        case AllocType::MANAGED: return "cudaMallocManaged";
        case AllocType::PINNED: return "cudaHostAlloc";
        case AllocType::MALLOC_REGISTERED: return "malloc+cudaHostRegister";
    }
    return "unknown";
}

static cudaError_t allocate_buffer(AllocType t, size_t bytes, float** dev_ptr, float** host_ptr)
{
    *dev_ptr = nullptr;
    if (host_ptr) *host_ptr = nullptr;
    switch (t) {
        case AllocType::DEVICE:
            return cudaMalloc((void**)dev_ptr, bytes);
        case AllocType::MANAGED:
            return cudaMallocManaged((void**)dev_ptr, bytes);
        case AllocType::PINNED:
            return cudaHostAlloc((void**)dev_ptr, bytes, cudaHostAllocDefault);
        case AllocType::MALLOC_REGISTERED: {
            float* h = (float*)malloc(bytes);
            if (!h) return cudaErrorMemoryAllocation;
            cudaError_t e = cudaHostRegister(h, bytes, cudaHostRegisterDefault);
            if (e != cudaSuccess) { free(h); return e; }
            *dev_ptr = h;
            if (host_ptr) *host_ptr = h;
            return cudaSuccess;
        }
    }
    return cudaErrorInvalidValue;
}

static void free_buffer(AllocType t, float* dev_ptr, float* host_ptr)
{
    if (!dev_ptr) return;
    if (t == AllocType::MALLOC_REGISTERED) {
        cudaHostUnregister(dev_ptr);
        free(host_ptr ? host_ptr : dev_ptr);
    } else if (t == AllocType::PINNED) {
        cudaFreeHost(dev_ptr);
    } else {
        cudaFree(dev_ptr);
    }
}

static bool safe_bytes(int64_t n, size_t* bytes)
{
    if (n < 0) return false;
    const size_t max_elems = std::numeric_limits<size_t>::max() / sizeof(float);
    if ((size_t)n > max_elems) return false;
    *bytes = (size_t)n * sizeof(float);
    return true;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ inline int64_t gtid(void)
{
    return (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
}

__device__ inline int64_t gsize(void)
{
    return (int64_t)gridDim.x * blockDim.x;
}

__global__ void sequential_read(const float* in, float* out, int64_t n)
{
    for (int64_t i = gtid(); i < n; i += gsize()) out[i] = in[i];
}

__global__ void sequential_write(const float* in, float* out, int64_t n)
{
    (void)in;
    for (int64_t i = gtid(); i < n; i += gsize()) out[i] = 1.0f;
}

__global__ void read_write_copy(const float* in, float* out, int64_t n)
{
    for (int64_t i = gtid(); i < n; i += gsize()) out[i] = in[i];
}

__global__ void saxpy(const float* x, float* y, float a, int64_t n)
{
    for (int64_t i = gtid(); i < n; i += gsize()) y[i] = a * x[i] + y[i];
}

__global__ void strided_read(const float* in, float* out, int64_t n, int stride)
{
    for (int64_t i = gtid(); i < n; i += gsize()) {
        int64_t idx = (i * (int64_t)stride) % n;
        out[i] = in[idx];
    }
}

__global__ void random_read(const float* in, float* out, const int64_t* indices,
                            int64_t index_count, int64_t n)
{
    for (int64_t i = gtid(); i < n; i += gsize()) {
        int64_t idx = indices[i % index_count];
        if (idx < 0 || idx >= n) idx = 0;
        out[i] = in[idx];
    }
}

__global__ void coalesced_access(const float* in, float* out, int64_t n)
{
    for (int64_t i = gtid(); i < n; i += gsize()) out[i] = in[i];
}

__global__ void non_coalesced_access(const float* in, float* out, int64_t n)
{
    for (int64_t i = gtid(); i < n; i += gsize()) {
        int64_t lane = i % 32;
        int64_t warpid = i / 32;
        int64_t col_size = (n + 31) / 32;
        int64_t idx = (lane * col_size + warpid) % n;
        out[i] = in[idx];
    }
}

__global__ void atomic_accumulate(const float* in, float* result, int64_t n)
{
    for (int64_t i = gtid(); i < n; i += gsize()) atomicAdd(result, in[i]);
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

static void cpu_sequential_read(const float* in, float* out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) out[i] = in[i];
}

static void cpu_sequential_write(const float* in, float* out, int64_t n)
{
    (void)in;
    for (int64_t i = 0; i < n; ++i) out[i] = 1.0f;
}

static void cpu_copy(const float* in, float* out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) out[i] = in[i];
}

static void cpu_saxpy(const float* x, float* y, float a, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

static void cpu_strided_read(const float* in, float* out, int64_t n, int stride)
{
    for (int64_t i = 0; i < n; ++i) {
        int64_t idx = (i * (int64_t)stride) % n;
        out[i] = in[idx];
    }
}

static void cpu_random_read(const float* in, float* out, const int64_t* indices,
                            int64_t index_count, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) {
        int64_t idx = indices[i % index_count];
        if (idx < 0 || idx >= n) idx = 0;
        out[i] = in[idx];
    }
}

static void cpu_coalesced(const float* in, float* out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) out[i] = in[i];
}

static void cpu_non_coalesced(const float* in, float* out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) {
        int64_t lane = i % 32;
        int64_t warpid = i / 32;
        int64_t col_size = (n + 31) / 32;
        int64_t idx = (lane * col_size + warpid) % n;
        out[i] = in[idx];
    }
}

static float cpu_atomic(const float* in, int64_t n)
{
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) s += in[i];
    return s;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fill(float* p, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) p[i] = (float)(i % 1024);
}

static std::vector<int64_t> make_indices(int64_t n, int64_t index_count)
{
    std::vector<int64_t> v(index_count);
    for (int64_t i = 0; i < index_count; ++i) v[i] = i % n;
    std::mt19937_64 rng(12345);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

static bool feq(float a, float b)
{
    if (std::isnan(a) || std::isnan(b)) return false;
    if (a == b) return true;
    return std::fabs(a - b) <= 1e-3f;
}

using Launcher = std::function<void(cudaStream_t)>;

static int choose_grid(int64_t n, int block, int max_sms)
{
    int max_blocks = max_sms * 4;
    int needed = (int)((n + block - 1) / block);
    if (needed < 1) needed = 1;
    if (needed > max_blocks) needed = max_blocks;
    return needed;
}

static int choose_iters(size_t bytes)
{
    if (bytes < 1024 * 1024) return 10;
    if (bytes < 256 * 1024 * 1024) return 5;
    return 3;
}

// ---------------------------------------------------------------------------
// Benchmark timing
// ---------------------------------------------------------------------------

static float run_benchmark(const char* name, int64_t n, size_t bytes, int block,
                           int grid, const Launcher& launch)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int w = 0; w < 2; ++w) {
        launch(0);
    }
    cudaDeviceSynchronize();

    int iters = choose_iters(bytes);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; ++i) {
        launch(0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float total_ms = ms / (float)iters;
    double seconds = total_ms / 1000.0;
    double bw = (double)bytes / seconds / 1e9;
    log_printf("  %-24s n=%12lld bytes=%12zu block=%4d grid=%5d iters=%d %.2f ms %.2f GB/s\n",
               name, (long long)n, bytes, block, grid, iters, total_ms, bw);
    return (float)bw;
}

// ---------------------------------------------------------------------------
// CPU contention thread
// ---------------------------------------------------------------------------

static void cpu_hammer(float* buf, int64_t n, std::atomic<bool>* run)
{
    while (run->load()) {
        for (int64_t i = 0; i < n; i += 1024) {
            buf[i % n] = (float)i;
        }
    }
}

// ---------------------------------------------------------------------------
// Verification (TDD-style: fail early if kernels are wrong)
// ---------------------------------------------------------------------------

static int test_kernels(int max_sms)
{
    const int64_t n = 1024;
    const int block = 256;
    const int grid = choose_grid(n, block, max_sms);
    size_t bytes = 0;
    safe_bytes(n, &bytes);

    float *d_in = nullptr, *d_out = nullptr, *d_x = nullptr, *d_y = nullptr, *d_result = nullptr;
    int64_t* d_idx = nullptr;
    float *h_in = (float*)malloc(bytes), *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    if (!h_in || !h_out || !h_ref) return -1;

    fill(h_in, n);
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    auto ok = [&](const char* name, float* out) -> bool {
        (void)name;
        cudaMemcpy(h_out, out, bytes, cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < n; ++i) {
            if (!feq(h_out[i], h_ref[i])) {
                log_printf("  FAIL %s at %lld: got %f expected %f\n",
                           name, (long long)i, h_out[i], h_ref[i]);
                return false;
            }
        }
        log_printf("  PASS %s\n", name);
        return true;
    };

    bool all = true;

    // sequential_read
    cudaMemset(d_out, 0, bytes);
    sequential_read<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpu_sequential_read(h_in, h_ref, n);
    all &= ok("sequential_read", d_out);

    // sequential_write
    cudaMemset(d_out, 0, bytes);
    sequential_write<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpu_sequential_write(h_in, h_ref, n);
    all &= ok("sequential_write", d_out);

    // read_write_copy
    cudaMemset(d_out, 0, bytes);
    read_write_copy<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpu_copy(h_in, h_ref, n);
    all &= ok("read_write_copy", d_out);

    // saxpy
    cudaMalloc((void**)&d_x, bytes);
    cudaMalloc((void**)&d_y, bytes);
    cudaMemcpy(d_x, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_in, bytes, cudaMemcpyHostToDevice);
    float a = 2.5f;
    saxpy<<<grid, block>>>(d_x, d_y, a, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_y, bytes, cudaMemcpyDeviceToHost);
    memcpy(h_ref, h_in, bytes);
    cpu_saxpy(h_in, h_ref, a, n);
    all &= ok("saxpy", d_y);

    // strided_read
    int stride = 8;
    cudaMemset(d_out, 0, bytes);
    strided_read<<<grid, block>>>(d_in, d_out, n, stride);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpu_strided_read(h_in, h_ref, n, stride);
    all &= ok("strided_read", d_out);

    // random_read
    int64_t index_count = n;
    auto idx = make_indices(n, index_count);
    cudaMalloc((void**)&d_idx, index_count * sizeof(int64_t));
    cudaMemcpy(d_idx, idx.data(), index_count * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, bytes);
    random_read<<<grid, block>>>(d_in, d_out, d_idx, index_count, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpu_random_read(h_in, h_ref, idx.data(), index_count, n);
    all &= ok("random_read", d_out);

    // coalesced_access
    cudaMemset(d_out, 0, bytes);
    coalesced_access<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpu_coalesced(h_in, h_ref, n);
    all &= ok("coalesced_access", d_out);

    // non_coalesced_access
    cudaMemset(d_out, 0, bytes);
    non_coalesced_access<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpu_non_coalesced(h_in, h_ref, n);
    all &= ok("non_coalesced_access", d_out);

    // atomic_accumulate
    cudaMalloc((void**)&d_result, sizeof(float));
    float h_result = 0.0f;
    cudaMemset(d_result, 0, sizeof(float));
    atomic_accumulate<<<grid, block>>>(d_in, d_result, n);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    float ref = cpu_atomic(h_in, n);
    if (!feq(h_result, ref)) {
        log_printf("  FAIL atomic_accumulate: got %f expected %f\n", h_result, ref);
        all = false;
    } else {
        log_printf("  PASS atomic_accumulate\n");
    }

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_x); cudaFree(d_y); cudaFree(d_result); cudaFree(d_idx);
    free(h_in); free(h_out); free(h_ref);
    return all ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Percentage of peak helpers
// ---------------------------------------------------------------------------

using LauncherFactory = std::function<Launcher(float*, float*)>;

static float percent_peak(float bw)
{
    return bw / 273.0f * 100.0f;
}

static bool safe_add_size(size_t* total, size_t add)
{
    if (*total > std::numeric_limits<size_t>::max() - add) return false;
    *total += add;
    return true;
}

static bool query_free_threshold(size_t* threshold)
{
    size_t free = 0, total = 0;
    cudaError_t e = cudaMemGetInfo(&free, &total);
    if (e != cudaSuccess) {
        log_printf("CUDA error: %s\n", cudaGetErrorString(e));
        return false;
    }
    *threshold = (size_t)((double)free * 0.8);
    return true;
}

// ---------------------------------------------------------------------------
// Working-set sweep helpers
// ---------------------------------------------------------------------------

static void workset_two_array(FILE* f, const char* name, int64_t n,
                              size_t arr_bytes, size_t traffic, int block,
                              int grid, size_t free_thresh,
                              const LauncherFactory& make_launch)
{
    size_t needed = 0;
    if (!safe_add_size(&needed, arr_bytes) || !safe_add_size(&needed, arr_bytes)) {
        fprintf(f, "%s,%zu,0.00,0.00,skip\n", name, arr_bytes);
        return;
    }
    if (needed > free_thresh) {
        fprintf(f, "%s,%zu,0.00,0.00,skip\n", name, arr_bytes);
        return;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc((void**)&d_in, arr_bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_out, arr_bytes) != cudaSuccess) {
        log_printf("  OOM %s at %zu bytes\n", name, arr_bytes);
        fprintf(f, "%s,%zu,0.00,0.00,oom\n", name, arr_bytes);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        return;
    }

    CUDA_CHECK(cudaMemset(d_in, 0, arr_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, arr_bytes));

    Launcher launch = make_launch(d_in, d_out);
    float bw = run_benchmark(name, n, traffic, block, grid, launch);
    float pct = percent_peak(bw);
    fprintf(f, "%s,%zu,%.2f,%.2f,\n", name, arr_bytes, bw, pct);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

static void workset_random(FILE* f, int64_t n, size_t arr_bytes, int block,
                           int grid, size_t free_thresh)
{
    const char* name = "random_read";
    int64_t index_count = std::min<int64_t>(n, 16LL * 1024 * 1024);
    size_t idx_bytes = (size_t)index_count * sizeof(int64_t);

    size_t needed = 0;
    if (!safe_add_size(&needed, arr_bytes) ||
        !safe_add_size(&needed, arr_bytes) ||
        !safe_add_size(&needed, idx_bytes)) {
        fprintf(f, "%s,%zu,0.00,0.00,skip\n", name, arr_bytes);
        return;
    }
    if (needed > free_thresh) {
        fprintf(f, "%s,%zu,0.00,0.00,skip\n", name, arr_bytes);
        return;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    int64_t* d_idx = nullptr;
    if (cudaMalloc((void**)&d_in, arr_bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_out, arr_bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_idx, idx_bytes) != cudaSuccess) {
        log_printf("  OOM %s at %zu bytes\n", name, arr_bytes);
        fprintf(f, "%s,%zu,0.00,0.00,oom\n", name, arr_bytes);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_idx));
        return;
    }

    auto idx = make_indices(n, index_count);
    CUDA_CHECK(cudaMemcpy(d_idx, idx.data(), idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_in, 0, arr_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, arr_bytes));

    Launcher launch = [=](cudaStream_t s) {
        random_read<<<grid, block, 0, s>>>(d_in, d_out, d_idx, index_count, n);
    };
    float bw = run_benchmark(name, n, needed, block, grid, launch);
    float pct = percent_peak(bw);
    fprintf(f, "%s,%zu,%.2f,%.2f,\n", name, arr_bytes, bw, pct);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_idx));
}

static void workset_atomic(FILE* f, int64_t n, size_t arr_bytes, int block,
                           int grid, size_t free_thresh)
{
    const char* name = "atomic_accumulate";
    size_t result_bytes = sizeof(float);

    size_t needed = 0;
    if (!safe_add_size(&needed, arr_bytes) || !safe_add_size(&needed, result_bytes)) {
        fprintf(f, "%s,%zu,0.00,0.00,skip\n", name, arr_bytes);
        return;
    }
    if (needed > free_thresh) {
        fprintf(f, "%s,%zu,0.00,0.00,skip\n", name, arr_bytes);
        return;
    }

    size_t traffic = 0;
    if (!safe_add_size(&traffic, arr_bytes) || !safe_add_size(&traffic, arr_bytes)) {
        fprintf(f, "%s,%zu,0.00,0.00,skip\n", name, arr_bytes);
        return;
    }

    float* d_in = nullptr;
    float* d_result = nullptr;
    if (cudaMalloc((void**)&d_in, arr_bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_result, result_bytes) != cudaSuccess) {
        log_printf("  OOM %s at %zu bytes\n", name, arr_bytes);
        fprintf(f, "%s,%zu,0.00,0.00,oom\n", name, arr_bytes);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_result));
        return;
    }

    CUDA_CHECK(cudaMemset(d_in, 0, arr_bytes));
    CUDA_CHECK(cudaMemset(d_result, 0, result_bytes));

    Launcher launch = [=](cudaStream_t s) {
        atomic_accumulate<<<grid, block, 0, s>>>(d_in, d_result, n);
    };
    float bw = run_benchmark(name, n, traffic, block, grid, launch);
    float pct = percent_peak(bw);
    fprintf(f, "%s,%zu,%.2f,%.2f,\n", name, arr_bytes, bw, pct);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_result));
}

// ---------------------------------------------------------------------------
// 1. Working-set sweep
// ---------------------------------------------------------------------------

static void sweep_workset(int max_sms)
{
    const char* path = "results/bandwidth_workset.csv";
    FILE* f = fopen(path, "w");
    if (!f) {
        log_printf("Warning: cannot open %s for writing\n", path);
        return;
    }
    fprintf(f, "kernel,size_bytes,bandwidth_gb_s,percent_peak,note\n");

    const size_t sizes[] = {
        1ULL << 10, 1ULL << 12, 1ULL << 14, 1ULL << 16,
        1ULL << 18, 1ULL << 20, 1ULL << 22, 1ULL << 24,
        1ULL << 26, 1ULL << 28, 1ULL << 30, 1ULL << 32,
        1ULL << 34, 1ULL << 36
    };
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int block = 256;

    for (int i = 0; i < num_sizes; ++i) {
        size_t size = sizes[i];
        int64_t n = (int64_t)(size / sizeof(float));
        size_t arr_bytes = 0;
        if (!safe_bytes(n, &arr_bytes)) {
            log_printf("  Overflow at working-set size %zu\n", size);
            continue;
        }

        size_t free_thresh = 0;
        if (!query_free_threshold(&free_thresh)) {
            log_printf("  Cannot query free memory; aborting workset sweep\n");
            break;
        }

        int grid = choose_grid(n, block, max_sms);
        log_printf("Workset size %zu (n=%lld, grid=%d)\n", size, (long long)n, grid);

        // sequential_read
        {
            size_t traffic = 0;
            safe_add_size(&traffic, arr_bytes);
            safe_add_size(&traffic, arr_bytes);
            workset_two_array(f, "sequential_read", n, arr_bytes, traffic, block,
                              grid, free_thresh,
                [&](float* in, float* out) -> Launcher {
                    return [=](cudaStream_t s) {
                        sequential_read<<<grid, block, 0, s>>>(in, out, n);
                    };
                });
        }

        // sequential_write
        {
            size_t traffic = arr_bytes;
            workset_two_array(f, "sequential_write", n, arr_bytes, traffic, block,
                              grid, free_thresh,
                [&](float* in, float* out) -> Launcher {
                    return [=](cudaStream_t s) {
                        sequential_write<<<grid, block, 0, s>>>(in, out, n);
                    };
                });
        }

        // read_write_copy
        {
            size_t traffic = 0;
            safe_add_size(&traffic, arr_bytes);
            safe_add_size(&traffic, arr_bytes);
            workset_two_array(f, "read_write_copy", n, arr_bytes, traffic, block,
                              grid, free_thresh,
                [&](float* in, float* out) -> Launcher {
                    return [=](cudaStream_t s) {
                        read_write_copy<<<grid, block, 0, s>>>(in, out, n);
                    };
                });
        }

        // saxpy
        {
            size_t traffic = 0;
            safe_add_size(&traffic, arr_bytes);
            safe_add_size(&traffic, arr_bytes);
            safe_add_size(&traffic, arr_bytes);
            workset_two_array(f, "saxpy", n, arr_bytes, traffic, block,
                              grid, free_thresh,
                [&](float* in, float* out) -> Launcher {
                    return [=](cudaStream_t s) {
                        saxpy<<<grid, block, 0, s>>>(in, out, 2.5f, n);
                    };
                });
        }

        // strided_read
        {
            size_t traffic = 0;
            safe_add_size(&traffic, arr_bytes);
            safe_add_size(&traffic, arr_bytes);
            workset_two_array(f, "strided_read", n, arr_bytes, traffic, block,
                              grid, free_thresh,
                [&](float* in, float* out) -> Launcher {
                    int stride = 8;
                    return [=](cudaStream_t s) {
                        strided_read<<<grid, block, 0, s>>>(in, out, n, stride);
                    };
                });
        }

        // random_read
        workset_random(f, n, arr_bytes, block, grid, free_thresh);

        // coalesced_access
        {
            size_t traffic = 0;
            safe_add_size(&traffic, arr_bytes);
            safe_add_size(&traffic, arr_bytes);
            workset_two_array(f, "coalesced_access", n, arr_bytes, traffic, block,
                              grid, free_thresh,
                [&](float* in, float* out) -> Launcher {
                    return [=](cudaStream_t s) {
                        coalesced_access<<<grid, block, 0, s>>>(in, out, n);
                    };
                });
        }

        // non_coalesced_access
        {
            size_t traffic = 0;
            safe_add_size(&traffic, arr_bytes);
            safe_add_size(&traffic, arr_bytes);
            workset_two_array(f, "non_coalesced_access", n, arr_bytes, traffic, block,
                              grid, free_thresh,
                [&](float* in, float* out) -> Launcher {
                    return [=](cudaStream_t s) {
                        non_coalesced_access<<<grid, block, 0, s>>>(in, out, n);
                    };
                });
        }

        // atomic_accumulate
        workset_atomic(f, n, arr_bytes, block, grid, free_thresh);
    }

    fclose(f);
}

// ---------------------------------------------------------------------------
// 2. Stride sweep
// ---------------------------------------------------------------------------

static void sweep_stride(int max_sms)
{
    const char* path = "results/bandwidth_stride.csv";
    FILE* f = fopen(path, "w");
    if (!f) {
        log_printf("Warning: cannot open %s for writing\n", path);
        return;
    }
    fprintf(f, "stride,bandwidth_gb_s,percent_peak\n");

    const int64_t n = (64LL * 1024 * 1024) / sizeof(float);
    size_t arr_bytes = 0;
    if (!safe_bytes(n, &arr_bytes)) {
        log_printf("Overflow in sweep_stride\n");
        fclose(f);
        return;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc((void**)&d_in, arr_bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_out, arr_bytes) != cudaSuccess) {
        log_printf("OOM in sweep_stride\n");
        fclose(f);
        return;
    }

    CUDA_CHECK(cudaMemset(d_in, 0, arr_bytes));

    const int block = 256;
    int grid = choose_grid(n, block, max_sms);
    size_t bytes = 0;
    safe_add_size(&bytes, arr_bytes);
    safe_add_size(&bytes, arr_bytes);

    const int strides[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int i = 0; i < 8; ++i) {
        int stride = strides[i];
        Launcher launch = [=](cudaStream_t s) {
            strided_read<<<grid, block, 0, s>>>(d_in, d_out, n, stride);
        };
        float bw = run_benchmark("strided_read", n, bytes, block, grid, launch);
        float pct = percent_peak(bw);
        fprintf(f, "%d,%.2f,%.2f\n", stride, bw, pct);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    fclose(f);
}

// ---------------------------------------------------------------------------
// 3. Block-size sweep
// ---------------------------------------------------------------------------

static void sweep_block(int max_sms)
{
    const char* path = "results/bandwidth_block.csv";
    FILE* f = fopen(path, "w");
    if (!f) {
        log_printf("Warning: cannot open %s for writing\n", path);
        return;
    }
    fprintf(f, "block_size,bandwidth_gb_s,percent_peak\n");

    const int64_t n = (64LL * 1024 * 1024) / sizeof(float);
    size_t arr_bytes = 0;
    if (!safe_bytes(n, &arr_bytes)) {
        log_printf("Overflow in sweep_block\n");
        fclose(f);
        return;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc((void**)&d_in, arr_bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_out, arr_bytes) != cudaSuccess) {
        log_printf("OOM in sweep_block\n");
        fclose(f);
        return;
    }

    CUDA_CHECK(cudaMemset(d_in, 0, arr_bytes));

    size_t bytes = 0;
    safe_add_size(&bytes, arr_bytes);
    safe_add_size(&bytes, arr_bytes);

    const int blocks[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 6; ++i) {
        int block = blocks[i];
        int grid = choose_grid(n, block, max_sms);
        Launcher launch = [=](cudaStream_t s) {
            sequential_read<<<grid, block, 0, s>>>(d_in, d_out, n);
        };
        float bw = run_benchmark("sequential_read", n, bytes, block, grid, launch);
        float pct = percent_peak(bw);
        fprintf(f, "%d,%.2f,%.2f\n", block, bw, pct);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    fclose(f);
}

// ---------------------------------------------------------------------------
// 4. Allocation-type sweep
// ---------------------------------------------------------------------------

static void sweep_alloc(int max_sms)
{
    const char* path = "results/bandwidth_alloc.csv";
    FILE* f = fopen(path, "w");
    if (!f) {
        log_printf("Warning: cannot open %s for writing\n", path);
        return;
    }
    fprintf(f, "alloc_type,access,bandwidth_gb_s,percent_peak\n");

    const int64_t n = (256LL * 1024 * 1024) / sizeof(float);
    size_t arr_bytes = 0;
    if (!safe_bytes(n, &arr_bytes)) {
        log_printf("Overflow in sweep_alloc\n");
        fclose(f);
        return;
    }

    int block = 256;
    int grid = choose_grid(n, block, max_sms);
    size_t bytes = 0;
    safe_add_size(&bytes, arr_bytes);
    safe_add_size(&bytes, arr_bytes);

    // DEVICE
    {
        float *in = nullptr, *out = nullptr;
        if (allocate_buffer(AllocType::DEVICE, arr_bytes, &in, nullptr) == cudaSuccess &&
            allocate_buffer(AllocType::DEVICE, arr_bytes, &out, nullptr) == cudaSuccess) {
            CUDA_CHECK(cudaMemset(in, 0, arr_bytes));
            CUDA_CHECK(cudaMemset(out, 0, arr_bytes));
            Launcher launch = [=](cudaStream_t s) {
                sequential_read<<<grid, block, 0, s>>>(in, out, n);
            };
            float bw = run_benchmark("sequential_read", n, bytes, block, grid, launch);
            fprintf(f, "%s,%s,%.2f,%.2f\n", alloc_name(AllocType::DEVICE), "device", bw, percent_peak(bw));
        } else {
            log_printf("  OOM %s in sweep_alloc\n", alloc_name(AllocType::DEVICE));
            fprintf(f, "%s,%s,0.00,0.00\n", alloc_name(AllocType::DEVICE), "device");
        }
        free_buffer(AllocType::DEVICE, in, nullptr);
        free_buffer(AllocType::DEVICE, out, nullptr);
    }

    // MANAGED first and second
    {
        float *in = nullptr, *out = nullptr;
        if (allocate_buffer(AllocType::MANAGED, arr_bytes, &in, nullptr) == cudaSuccess &&
            allocate_buffer(AllocType::MANAGED, arr_bytes, &out, nullptr) == cudaSuccess) {
            CUDA_CHECK(cudaMemset(in, 0, arr_bytes));
            CUDA_CHECK(cudaMemset(out, 0, arr_bytes));
            CUDA_CHECK(cudaDeviceSynchronize());

            // One untimed launch to warm/page-fault
            sequential_read<<<grid, block>>>(in, out, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            // First timed launch
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start, 0));
            sequential_read<<<grid, block>>>(in, out, n);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            double seconds = (double)ms / 1000.0;
            double bw_d = (double)bytes / seconds / 1e9;
            float bw1 = (float)bw_d;
            log_printf("  %-24s n=%12lld bytes=%12zu block=%4d grid=%5d iters=1 %.2f ms %.2f GB/s (managed first)\n",
                       "sequential_read", (long long)n, bytes, block, grid, ms, bw1);
            fprintf(f, "%s,%s,%.2f,%.2f\n", alloc_name(AllocType::MANAGED), "first", bw1, percent_peak(bw1));

            // Second pass on warmed buffer
            Launcher launch = [=](cudaStream_t s) {
                sequential_read<<<grid, block, 0, s>>>(in, out, n);
            };
            float bw2 = run_benchmark("sequential_read", n, bytes, block, grid, launch);
            fprintf(f, "%s,%s,%.2f,%.2f\n", alloc_name(AllocType::MANAGED), "second", bw2, percent_peak(bw2));
        } else {
            log_printf("  OOM %s in sweep_alloc\n", alloc_name(AllocType::MANAGED));
            fprintf(f, "%s,%s,0.00,0.00\n", alloc_name(AllocType::MANAGED), "first");
            fprintf(f, "%s,%s,0.00,0.00\n", alloc_name(AllocType::MANAGED), "second");
        }
        free_buffer(AllocType::MANAGED, in, nullptr);
        free_buffer(AllocType::MANAGED, out, nullptr);
    }

    // PINNED
    {
        float *in = nullptr, *out = nullptr, *in_h = nullptr, *out_h = nullptr;
        if (allocate_buffer(AllocType::PINNED, arr_bytes, &in, &in_h) == cudaSuccess &&
            allocate_buffer(AllocType::PINNED, arr_bytes, &out, &out_h) == cudaSuccess) {
            CUDA_CHECK(cudaMemset(in, 0, arr_bytes));
            CUDA_CHECK(cudaMemset(out, 0, arr_bytes));
            Launcher launch = [=](cudaStream_t s) {
                sequential_read<<<grid, block, 0, s>>>(in, out, n);
            };
            float bw = run_benchmark("sequential_read", n, bytes, block, grid, launch);
            fprintf(f, "%s,%s,%.2f,%.2f\n", alloc_name(AllocType::PINNED), "device", bw, percent_peak(bw));
        } else {
            log_printf("  OOM %s in sweep_alloc\n", alloc_name(AllocType::PINNED));
            fprintf(f, "%s,%s,0.00,0.00\n", alloc_name(AllocType::PINNED), "device");
        }
        free_buffer(AllocType::PINNED, in, in_h);
        free_buffer(AllocType::PINNED, out, out_h);
    }

    // MALLOC_REGISTERED
    {
        float *in = nullptr, *out = nullptr, *in_h = nullptr, *out_h = nullptr;
        if (allocate_buffer(AllocType::MALLOC_REGISTERED, arr_bytes, &in, &in_h) == cudaSuccess &&
            allocate_buffer(AllocType::MALLOC_REGISTERED, arr_bytes, &out, &out_h) == cudaSuccess) {
            CUDA_CHECK(cudaMemset(in, 0, arr_bytes));
            CUDA_CHECK(cudaMemset(out, 0, arr_bytes));
            Launcher launch = [=](cudaStream_t s) {
                sequential_read<<<grid, block, 0, s>>>(in, out, n);
            };
            float bw = run_benchmark("sequential_read", n, bytes, block, grid, launch);
            fprintf(f, "%s,%s,%.2f,%.2f\n", alloc_name(AllocType::MALLOC_REGISTERED), "device", bw, percent_peak(bw));
        } else {
            log_printf("  OOM %s in sweep_alloc\n", alloc_name(AllocType::MALLOC_REGISTERED));
            fprintf(f, "%s,%s,0.00,0.00\n", alloc_name(AllocType::MALLOC_REGISTERED), "device");
        }
        free_buffer(AllocType::MALLOC_REGISTERED, in, in_h);
        free_buffer(AllocType::MALLOC_REGISTERED, out, out_h);
    }

    fclose(f);
}

// ---------------------------------------------------------------------------
// 5. UMA contention sweep
// ---------------------------------------------------------------------------

static void sweep_uma(int max_sms)
{
    const char* path = "results/bandwidth_uma.csv";
    FILE* f = fopen(path, "w");
    if (!f) {
        log_printf("Warning: cannot open %s for writing\n", path);
        return;
    }
    fprintf(f, "alloc_type,contention,bandwidth_gb_s,percent_peak\n");

    const int64_t n = (256LL * 1024 * 1024) / sizeof(float);
    size_t arr_bytes = 0;
    if (!safe_bytes(n, &arr_bytes)) {
        log_printf("Overflow in sweep_uma\n");
        fclose(f);
        return;
    }

    int block = 256;
    int grid = choose_grid(n, block, max_sms);
    size_t bytes = 0;
    safe_add_size(&bytes, arr_bytes);
    safe_add_size(&bytes, arr_bytes);

    int64_t cpu_n = n;
    size_t cpu_bytes = arr_bytes;

    struct Cond {
        AllocType type;
        const char* contention;
    };
    const Cond conds[] = {
        {AllocType::DEVICE, "none"},
        {AllocType::DEVICE, "cpu"},
        {AllocType::MANAGED, "none"},
        {AllocType::MANAGED, "cpu"},
    };

    for (size_t ci = 0; ci < sizeof(conds) / sizeof(conds[0]); ++ci) {
        AllocType type = conds[ci].type;
        const char* contention = conds[ci].contention;
        bool cpu = std::strcmp(contention, "cpu") == 0;

        float *in = nullptr, *out = nullptr;
        float *in_h = nullptr, *out_h = nullptr;
        float *cpu_buf = nullptr, *cpu_buf_h = nullptr;
        std::thread* t = nullptr;
        std::atomic<bool> run{false};
        bool oom = false;

        if (allocate_buffer(type, arr_bytes, &in, &in_h) != cudaSuccess) oom = true;
        if (!oom && allocate_buffer(type, arr_bytes, &out, &out_h) != cudaSuccess) oom = true;

        if (cpu && !oom) {
            if (allocate_buffer(AllocType::MANAGED, cpu_bytes, &cpu_buf, &cpu_buf_h) != cudaSuccess) {
                oom = true;
            } else {
                run.store(true);
                t = new std::thread(cpu_hammer, cpu_buf, cpu_n, &run);
            }
        }

        if (!oom) {
            CUDA_CHECK(cudaMemset(in, 0, arr_bytes));
            Launcher launch = [=](cudaStream_t s) {
                sequential_read<<<grid, block, 0, s>>>(in, out, n);
            };
            float bw = run_benchmark("sequential_read", n, bytes, block, grid, launch);
            float pct = percent_peak(bw);
            fprintf(f, "%s,%s,%.2f,%.2f\n", alloc_name(type), contention, bw, pct);
        } else {
            log_printf("  OOM %s %s in sweep_uma\n", alloc_name(type), contention);
            fprintf(f, "%s,%s,0.00,0.00\n", alloc_name(type), contention);
        }

        if (t) {
            run.store(false);
            t->join();
            delete t;
        }

        free_buffer(type, in, in_h);
        free_buffer(type, out, out_h);
        free_buffer(AllocType::MANAGED, cpu_buf, cpu_buf_h);
    }

    fclose(f);
}

// ---------------------------------------------------------------------------
// 6. Nsight Compute launch helper
// ---------------------------------------------------------------------------

static bool is_valid_kernel_name(const char* s)
{
    if (!s || s[0] == '\0' || s[0] == '-') return false;
    if (std::strstr(s, "..") != nullptr) return false;
    if (std::strchr(s, '/') != nullptr || std::strchr(s, '\\') != nullptr) return false;
    const char* allowed[] = {
        "sequential_read", "sequential_write", "read_write_copy",
        "saxpy", "strided_read", "random_read",
        "coalesced_access", "non_coalesced_access", "atomic_accumulate"
    };
    for (size_t i = 0; i < sizeof(allowed) / sizeof(allowed[0]); ++i) {
        if (std::strcmp(s, allowed[i]) == 0) return true;
    }
    return false;
}

static void ncu_run(int max_sms, const char* kernel_name)
{
    if (!is_valid_kernel_name(kernel_name)) {
        log_printf("Invalid ncu kernel name: %s\n", kernel_name);
        return;
    }

    const int64_t n = (256LL * 1024 * 1024) / sizeof(float);
    size_t arr_bytes = 0;
    if (!safe_bytes(n, &arr_bytes)) {
        log_printf("Overflow in ncu_run\n");
        return;
    }

    int block = 256;
    int grid = choose_grid(n, block, max_sms);
    log_printf("ncu_run: %s n=%lld grid=%d block=%d\n",
               kernel_name, (long long)n, grid, block);

    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_result = nullptr;
    int64_t* d_idx = nullptr;

    if (std::strcmp(kernel_name, "saxpy") == 0) {
        if (cudaMalloc((void**)&d_x, arr_bytes) == cudaSuccess &&
            cudaMalloc((void**)&d_y, arr_bytes) == cudaSuccess) {
            CUDA_CHECK(cudaMemset(d_x, 0, arr_bytes));
            CUDA_CHECK(cudaMemset(d_y, 0, arr_bytes));
            saxpy<<<grid, block>>>(d_x, d_y, 2.5f, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            log_printf("ncu_run: %s done\n", kernel_name);
        } else {
            log_printf("OOM in ncu_run for %s\n", kernel_name);
        }
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    } else if (std::strcmp(kernel_name, "atomic_accumulate") == 0) {
        if (cudaMalloc((void**)&d_in, arr_bytes) == cudaSuccess &&
            cudaMalloc((void**)&d_result, sizeof(float)) == cudaSuccess) {
            CUDA_CHECK(cudaMemset(d_in, 0, arr_bytes));
            CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
            atomic_accumulate<<<grid, block>>>(d_in, d_result, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            log_printf("ncu_run: %s done\n", kernel_name);
        } else {
            log_printf("OOM in ncu_run for %s\n", kernel_name);
        }
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_result));
    } else if (std::strcmp(kernel_name, "random_read") == 0) {
        if (cudaMalloc((void**)&d_in, arr_bytes) == cudaSuccess &&
            cudaMalloc((void**)&d_out, arr_bytes) == cudaSuccess) {
            int64_t index_count = std::min<int64_t>(n, 16LL * 1024 * 1024);
            size_t idx_bytes = (size_t)index_count * sizeof(int64_t);
            if (cudaMalloc((void**)&d_idx, idx_bytes) == cudaSuccess) {
                auto idx = make_indices(n, index_count);
                CUDA_CHECK(cudaMemcpy(d_idx, idx.data(), idx_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_in, 0, arr_bytes));
                random_read<<<grid, block>>>(d_in, d_out, d_idx, index_count, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                log_printf("ncu_run: %s done\n", kernel_name);
            } else {
                log_printf("OOM in ncu_run for %s index buffer\n", kernel_name);
            }
            CUDA_CHECK(cudaFree(d_idx));
        } else {
            log_printf("OOM in ncu_run for %s\n", kernel_name);
        }
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    } else {
        // Kernels with two array arguments: sequential_read, sequential_write,
        // read_write_copy, strided_read, coalesced_access, non_coalesced_access
        if (cudaMalloc((void**)&d_in, arr_bytes) == cudaSuccess &&
            cudaMalloc((void**)&d_out, arr_bytes) == cudaSuccess) {
            CUDA_CHECK(cudaMemset(d_in, 0, arr_bytes));
            if (std::strcmp(kernel_name, "sequential_read") == 0) {
                sequential_read<<<grid, block>>>(d_in, d_out, n);
            } else if (std::strcmp(kernel_name, "sequential_write") == 0) {
                sequential_write<<<grid, block>>>(d_in, d_out, n);
            } else if (std::strcmp(kernel_name, "read_write_copy") == 0) {
                read_write_copy<<<grid, block>>>(d_in, d_out, n);
            } else if (std::strcmp(kernel_name, "strided_read") == 0) {
                strided_read<<<grid, block>>>(d_in, d_out, n, 8);
            } else if (std::strcmp(kernel_name, "coalesced_access") == 0) {
                coalesced_access<<<grid, block>>>(d_in, d_out, n);
            } else if (std::strcmp(kernel_name, "non_coalesced_access") == 0) {
                non_coalesced_access<<<grid, block>>>(d_in, d_out, n);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            log_printf("ncu_run: %s done\n", kernel_name);
        } else {
            log_printf("OOM in ncu_run for %s\n", kernel_name);
        }
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
}

// ---------------------------------------------------------------------------
// 7. Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::error_code ec;
    std::filesystem::create_directories("results", ec);
    if (ec) {
        printf("Warning: cannot create results directory: %s\n", ec.message().c_str());
    }

    FILE* log = fopen(LOG_PATH, "w");
    if (log) {
        g_log = log;
    } else {
        printf("Warning: cannot open %s; logging to stdout only\n", LOG_PATH);
    }

    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) {
        log_printf("CUDA error: %s\n", cudaGetErrorString(e));
        if (g_log) { fclose(g_log); g_log = nullptr; }
        return 1;
    }
    int max_sms = prop.multiProcessorCount;
    log_printf("Device: %s SMs: %d Peak: 273 GB/s\n", prop.name, max_sms);

    bool do_test = false;
    bool do_workset = false, do_stride = false, do_block = false;
    bool do_alloc = false, do_uma = false;
    bool do_run = false, do_all = false;
    bool do_ncu_one = false, do_ncu_all = false;
    std::string ncu_kernel;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--test") == 0) do_test = true;
        else if (std::strcmp(a, "--workset") == 0) do_workset = true;
        else if (std::strcmp(a, "--stride") == 0) do_stride = true;
        else if (std::strcmp(a, "--block") == 0) do_block = true;
        else if (std::strcmp(a, "--alloc") == 0) do_alloc = true;
        else if (std::strcmp(a, "--uma") == 0) do_uma = true;
        else if (std::strcmp(a, "--run") == 0) do_run = true;
        else if (std::strcmp(a, "--all") == 0) do_all = true;
        else if (std::strcmp(a, "--ncu") == 0) {
            if (i + 1 >= argc) {
                log_printf("Error: --ncu requires a kernel name\n");
                if (g_log) { fclose(g_log); g_log = nullptr; }
                return 1;
            }
            do_ncu_one = true;
            ncu_kernel = argv[++i];
        } else {
            log_printf("Unknown argument: %s\n", a);
            if (g_log) { fclose(g_log); g_log = nullptr; }
            return 1;
        }
    }

    if (do_run) {
        do_workset = true; do_stride = true; do_block = true;
        do_alloc = true; do_uma = true;
    }
    if (do_all) {
        do_test = true;
        do_workset = true; do_stride = true; do_block = true;
        do_alloc = true; do_uma = true;
        do_ncu_all = true;
    }
    if (argc == 1) {
        do_test = true;
        do_workset = true; do_stride = true; do_block = true;
        do_alloc = true; do_uma = true;
    }

    if (do_test) {
        log_printf("Running kernel tests...\n");
        if (test_kernels(max_sms) != 0) {
            log_printf("Kernel tests FAILED\n");
            if (g_log) { fclose(g_log); g_log = nullptr; }
            return 1;
        }
        log_printf("Kernel tests PASSED\n");
    }

    if (do_workset) sweep_workset(max_sms);
    if (do_stride) sweep_stride(max_sms);
    if (do_block) sweep_block(max_sms);
    if (do_alloc) sweep_alloc(max_sms);
    if (do_uma) sweep_uma(max_sms);

    if (do_ncu_one) {
        if (!is_valid_kernel_name(ncu_kernel.c_str())) {
            log_printf("Error: invalid ncu kernel name: %s\n", ncu_kernel.c_str());
            if (g_log) { fclose(g_log); g_log = nullptr; }
            return 1;
        }
        ncu_run(max_sms, ncu_kernel.c_str());
    }
    if (do_ncu_all) {
        const char* ncu_kernels[] = {"sequential_read", "read_write_copy", "saxpy"};
        for (size_t i = 0; i < sizeof(ncu_kernels) / sizeof(ncu_kernels[0]); ++i) {
            ncu_run(max_sms, ncu_kernels[i]);
        }
    }

    if (g_log) {
        fclose(g_log);
        g_log = nullptr;
    }
    return 0;
}

