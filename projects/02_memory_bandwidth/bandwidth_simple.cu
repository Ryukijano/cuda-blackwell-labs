// Project 02: Memory Bandwidth & Latency Lab
// Phase 1 — Hardware and Memory Literacy
//
// Implements a suite of memory access kernels and measures effective
// bandwidth across different patterns, working-set sizes, strides, and
// allocation types on the GB10 DGX Spark.
//
// Build:  make
// Run:    make run
// Profile: make profile

#include "cuda_utils.h"
#include "benchmark.h"
#include <vector>
#include <algorithm>
#include <thread>
#include <unistd.h>

// ============================================================================
// Kernel Definitions
// ============================================================================

// Sequential read: each thread reads one element, writes a dummy output
// Bytes moved: n * 4 (read) + n * 4 (write) = 8n bytes
__global__ void sequential_read(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// Sequential write: each thread writes one element
// Bytes moved: n * 4 (write) = 4n bytes
__global__ void sequential_write(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f;
    }
}

// Read-write-copy: read from in, write to out (like a memcpy)
// Bytes moved: n * 4 (read) + n * 4 (write) = 8n bytes
__global__ void read_write_copy(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

// SAXPY: y = a*x + y
// Bytes moved: n * 4 (read x) + n * 4 (read y) + n * 4 (write y) = 12n bytes
__global__ void saxpy_kernel(const float* x, float* y, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

// Strided read: each thread reads every `stride` elements
// Bytes moved: (n/stride) * 4 (read) + (n/stride) * 4 (write) = 8n/stride bytes
__global__ void strided_read(const float* in, float* out, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        out[idx / stride] = in[idx];
    }
}

// Random read: each thread reads from a random index
// Bytes moved: n * 4 (read indices) + n * 4 (read data) + n * 4 (write) = 12n bytes
__global__ void random_read(const float* in, float* out, const int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[indices[idx]];
    }
}

// Coalesced access: threads in a warp access consecutive addresses
// This is the same as sequential_read but explicit
__global__ void coalesced_access(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + 1.0f;
    }
}

// Non-coalesced access: threads in a warp access different cache lines
// Uses column-major access pattern to break coalescing
// Each thread in a warp reads from a stride of (n/32) elements apart
__global__ void non_coalesced_access(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int row = tid / 32;
        int col = tid % 32;
        // Column-major access: each thread in warp hits a different cache line
        int idx = col * ((n + 31) / 32) + row;
        if (idx < n) {
            out[tid] = in[idx];
        }
    }
}

// Atomic accumulate: all threads atomically add to a single result
// This demonstrates atomic contention
__global__ void atomic_accumulate(const float* in, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, in[idx]);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void init_array(float* arr, int n, float val = 1.0f) {
    for (int i = 0; i < n; i++) arr[i] = val;
}

void init_indices(int* indices, int n, unsigned seed = 42) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        indices[i] = rand() % n;
    }
}

// ============================================================================
// Part B: Bandwidth measurement for a single configuration
// ============================================================================

void measure_kernel(const char* name, int n, int block_size,
                    void (*launch_fn)(int, int, int)) {
    int bytes = n * sizeof(float);
    int grid = (n + block_size - 1) / block_size;

    // Warmup
    for (int i = 0; i < 3; i++) launch_fn(grid, block_size, n);
    cudaDeviceSynchronize();

    // Measure
    int iters = 10;
    GpuTimer timer;
    std::vector<double> times(iters);
    for (int i = 0; i < iters; i++) {
        timer.start();
        launch_fn(grid, block_size, n);
        timer.stop();
        times[i] = timer.milliseconds();
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / iters;
    double bw = effective_bandwidth_gbps(bytes * 2, mean); // read + write

    printf("  %-25s  n=%8d  block=%4d  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
           name, n, block_size, mean, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
}

// ============================================================================
// Part C: Working-set size sweep
// ============================================================================

void sweep_working_set() {
    print_header("Working-Set Size Sweep (block_size=256)");

    // Sizes: 1KB to 16GB (in elements)
    // 1KB = 256 floats, 4KB = 1024, 16KB = 4096, 64KB = 16384,
    // 256KB = 65536, 1MB = 262144, 4MB = 1048576, 16MB = 4194304,
    // 64MB = 16777216, 256MB = 67108864, 1GB = 268435456, 4GB = 1073741824
    size_t sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576,
                      4194304, 16777216, 67108864, 268435456};
    const char* size_labels[] = {"1KB", "4KB", "16KB", "64KB", "256KB", "1MB",
                                  "4MB", "16MB", "64MB", "256MB", "1GB"};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int block_size = 256;

    printf("  %-8s  %12s  %12s  %12s  %12s\n",
           "Size", "SeqRead", "SeqWrite", "Copy", "SAXPY");
    printf("  %-8s  %12s  %12s  %12s  %12s\n",
           "------", "----------", "----------", "----------", "----------");

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        size_t bytes = n * sizeof(float);

        // Allocate
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        cudaMemset(d_in, 1, bytes);
        cudaMemset(d_out, 0, bytes);

        int grid = (n + block_size - 1) / block_size;
        int iters = 10;
        int warmup = 3;

        auto run_kernel = [&](auto kernel, size_t bytes_moved) -> double {
            for (int i = 0; i < warmup; i++) kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
            cudaDeviceSynchronize();
            GpuTimer timer;
            std::vector<double> times(iters);
            for (int i = 0; i < iters; i++) {
                timer.start();
                kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
                timer.stop();
                times[i] = timer.milliseconds();
            }
            double mean = std::accumulate(times.begin(), times.end(), 0.0) / iters;
            return effective_bandwidth_gbps(bytes_moved, mean);
        };

        double bw_read   = run_kernel(sequential_read,   bytes * 2);  // read + write
        double bw_write  = run_kernel(sequential_write,  bytes);      // write only
        double bw_copy   = run_kernel(read_write_copy,   bytes * 2);  // read + write

        // SAXPY: measure separately since it has different signature
        double bw_saxpy;
        {
            for (int i = 0; i < warmup; i++) saxpy_kernel<<<grid, block_size>>>(d_in, d_out, 2.0f, (int)n);
            cudaDeviceSynchronize();
            GpuTimer timer;
            std::vector<double> times(iters);
            for (int i = 0; i < iters; i++) {
                timer.start();
                saxpy_kernel<<<grid, block_size>>>(d_in, d_out, 2.0f, (int)n);
                timer.stop();
                times[i] = timer.milliseconds();
            }
            double mean = std::accumulate(times.begin(), times.end(), 0.0) / iters;
            bw_saxpy = effective_bandwidth_gbps(bytes * 3, mean);  // read x + read y + write y
        }

        printf("  %-8s  %10.1f    %10.1f    %10.1f    %10.1f\n",
               size_labels[s], bw_read, bw_write, bw_copy, bw_saxpy);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
}

// ============================================================================
// Part C2: Stride sweep
// ============================================================================

void sweep_stride() {
    print_header("Stride Sweep (64MB working set, block_size=256)");

    size_t n = 16777216;  // 64MB of floats
    size_t bytes = n * sizeof(float);
    int block_size = 256;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaMemset(d_in, 1, bytes);

    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_strides = sizeof(strides) / sizeof(strides[0]);

    printf("  %-8s  %12s  %12s  %12s\n", "Stride", "BW (GB/s)", "% Peak", "Elements");
    printf("  %-8s  %12s  %12s  %12s\n", "------", "----------", "------", "--------");

    for (int s = 0; s < num_strides; s++) {
        int stride = strides[s];
        int effective_n = n / stride;
        int grid = (effective_n + block_size - 1) / block_size;
        size_t bytes_moved = effective_n * sizeof(float) * 2;  // read + write

        // Warmup
        for (int i = 0; i < 3; i++)
            strided_read<<<grid, block_size>>>(d_in, d_out, (int)n, stride);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            strided_read<<<grid, block_size>>>(d_in, d_out, (int)n, stride);
            timer.stop();
            times[i] = timer.milliseconds();
        }

        double mean = std::accumulate(times.begin(), times.end(), 0.0) / 10;
        double bw = effective_bandwidth_gbps(bytes_moved, mean);

        printf("  %-8d  %10.1f    %10.1f%%  %12d\n",
               stride, bw, 100.0 * bw / GB10_PEAK_BW_GBPS, effective_n);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// Part C3: Block size sweep
// ============================================================================

void sweep_block_size() {
    print_header("Block Size Sweep (64MB working set)");

    size_t n = 16777216;  // 64MB of floats
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaMemset(d_in, 1, bytes);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);

    printf("  %-12s  %12s  %12s  %12s\n", "Block Size", "BW (GB/s)", "% Peak", "Grid Size");
    printf("  %-12s  %12s  %12s  %12s\n", "----------", "----------", "------", "---------");

    for (int b = 0; b < num_blocks; b++) {
        int bs = block_sizes[b];
        int grid = (n + bs - 1) / bs;

        // Warmup
        for (int i = 0; i < 3; i++)
            sequential_read<<<grid, bs>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            sequential_read<<<grid, bs>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }

        double mean = std::accumulate(times.begin(), times.end(), 0.0) / 10;
        double bw = effective_bandwidth_gbps(bytes * 2, mean);

        printf("  %-12d  %10.1f    %10.1f%%  %12d\n",
               bs, bw, 100.0 * bw / GB10_PEAK_BW_GBPS, grid);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// Part D: Coalesced vs Non-coalesced
// ============================================================================

void compare_coalesced() {
    print_header("Coalesced vs Non-Coalesced Access (64MB)");

    size_t n = 16777216;  // 64MB
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaMemset(d_in, 1, bytes);

    auto run = [&](auto kernel, const char* name) {
        for (int i = 0; i < 3; i++) kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / 10;
        double bw = effective_bandwidth_gbps(bytes * 2, mean);
        printf("  %-25s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
               name, mean, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
    };

    run(coalesced_access, "coalesced_access");
    run(non_coalesced_access, "non_coalesced_access");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// Part E: Random read vs sequential read
// ============================================================================

void compare_random() {
    print_header("Random Read vs Sequential Read (64MB)");

    size_t n = 16777216;  // 64MB
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    float *d_in, *d_out;
    int *d_indices;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_indices, n * sizeof(int)));
    cudaMemset(d_in, 1, bytes);

    // Initialize random indices on host and copy to device
    std::vector<int> h_indices(n);
    init_indices(h_indices.data(), n);
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    auto run = [&](auto kernel, const char* name, size_t bytes_moved) {
        for (int i = 0; i < 3; i++) kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / 10;
        double bw = effective_bandwidth_gbps(bytes_moved, mean);
        printf("  %-25s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
               name, mean, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
    };

    // Sequential: read n floats + write n floats = 8n bytes
    run(sequential_read, "sequential_read", bytes * 2);

    // Random: read n indices + read n floats + write n floats = 12n bytes
    auto random_launch = [&]() {
        random_read<<<grid, block_size>>>(d_in, d_out, d_indices, (int)n);
    };
    for (int i = 0; i < 3; i++) random_launch();
    cudaDeviceSynchronize();
    GpuTimer timer;
    std::vector<double> times(10);
    for (int i = 0; i < 10; i++) {
        timer.start();
        random_launch();
        timer.stop();
        times[i] = timer.milliseconds();
    }
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / 10;
    double bw = effective_bandwidth_gbps(bytes * 3, mean);
    printf("  %-25s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
           "random_read", mean, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_indices));
}

// ============================================================================
// Part F: Atomic accumulate
// ============================================================================

void test_atomic() {
    print_header("Atomic Accumulation (64MB)");

    size_t n = 16777216;  // 64MB
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    float *d_in, *d_result;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    cudaMemset(d_in, 1, bytes);

    // Warmup
    float zero = 0.0f;
    for (int i = 0; i < 3; i++) {
        cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
        atomic_accumulate<<<grid, block_size>>>(d_in, d_result, (int)n);
    }
    cudaDeviceSynchronize();

    GpuTimer timer;
    std::vector<double> times(10);
    for (int i = 0; i < 10; i++) {
        cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
        timer.start();
        atomic_accumulate<<<grid, block_size>>>(d_in, d_result, (int)n);
        timer.stop();
        times[i] = timer.milliseconds();
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / 10;
    double bw = effective_bandwidth_gbps(bytes, mean);  // read only
    printf("  %-25s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
           "atomic_accumulate", mean, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
    printf("  (Low bandwidth expected — atomic contention serializes writes)\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_result));
}

// ============================================================================
// Part G: Allocation type comparison
// ============================================================================

void compare_alloc_types() {
    print_header("Allocation Type Comparison (256MB sequential read)");

    size_t n = 67108864;  // 256MB of floats
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    auto run_bench = [&](float* d_in, float* d_out, const char* name) {
        cudaMemset(d_in, 1, bytes);
        // Warmup
        for (int i = 0; i < 3; i++) sequential_read<<<grid, block_size>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            sequential_read<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / 10;
        double bw = effective_bandwidth_gbps(bytes * 2, mean);
        printf("  %-25s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
               name, mean, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
    };

    // 1. cudaMalloc
    {
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        run_bench(d_in, d_out, "cudaMalloc");
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    // 2. cudaMallocManaged
    {
        float *d_in, *d_out;
        CUDA_CHECK(cudaMallocManaged(&d_in, bytes));
        CUDA_CHECK(cudaMallocManaged(&d_out, bytes));
        // First access triggers page faults — measure both
        printf("\n  cudaMallocManaged (first access):\n");
        {
            GpuTimer timer;
            cudaMemset(d_in, 1, bytes);  // trigger migration
            timer.start();
            sequential_read<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            double bw = effective_bandwidth_gbps(bytes * 2, timer.milliseconds());
            printf("  %-25s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
                   "  managed (first)", timer.milliseconds(), bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
        }
        printf("  cudaMallocManaged (second access):\n");
        run_bench(d_in, d_out, "  managed (warm)");
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    // 3. cudaHostAlloc (pinned)
    {
        float *h_in, *h_out;
        CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&h_out, bytes, cudaHostAllocMapped));
        // Get device pointers
        float *d_in, *d_out;
        CUDA_CHECK(cudaHostGetDevicePointer(&d_in, h_in, 0));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_out, h_out, 0));
        memset(h_in, 1, bytes);
        run_bench(d_in, d_out, "cudaHostAlloc (pinned)");
        CUDA_CHECK(cudaFreeHost(h_in));
        CUDA_CHECK(cudaFreeHost(h_out));
    }
}

// ============================================================================
// Part H: UMA contention experiment
// ============================================================================

void uma_contention() {
    print_header("UMA Contention: GPU-only vs GPU+CPU (256MB)");

    size_t n = 67108864;  // 256MB
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaMemset(d_in, 1, bytes);

    // GPU-only baseline
    for (int i = 0; i < 3; i++) sequential_read<<<grid, block_size>>>(d_in, d_out, (int)n);
    cudaDeviceSynchronize();

    GpuTimer timer;
    std::vector<double> times(10);
    for (int i = 0; i < 10; i++) {
        timer.start();
        sequential_read<<<grid, block_size>>>(d_in, d_out, (int)n);
        timer.stop();
        times[i] = timer.milliseconds();
    }
    double gpu_only_ms = std::accumulate(times.begin(), times.end(), 0.0) / 10;
    double gpu_only_bw = effective_bandwidth_gbps(bytes * 2, gpu_only_ms);
    printf("  GPU only:     time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
           gpu_only_ms, gpu_only_bw, 100.0 * gpu_only_bw / GB10_PEAK_BW_GBPS);

    // GPU + CPU contention
    // Allocate host buffer for CPU to hammer
    float* h_hammer = (float*)malloc(bytes);
    memset(h_hammer, 0, bytes);

    // Launch CPU thread that reads/writes memory continuously
    volatile bool cpu_running = true;
    auto cpu_thread = std::thread([&]() {
        volatile float sum = 0;
        while (cpu_running) {
            for (size_t i = 0; i < n; i++) {
                sum += h_hammer[i];
                h_hammer[i] = sum;
            }
        }
        (void)sum;
    });

    // Wait a moment for CPU thread to start
    usleep(100000);  // 100ms

    // Measure GPU bandwidth while CPU is hammering memory
    for (int i = 0; i < 3; i++) sequential_read<<<grid, block_size>>>(d_in, d_out, (int)n);
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++) {
        timer.start();
        sequential_read<<<grid, block_size>>>(d_in, d_out, (int)n);
        timer.stop();
        times[i] = timer.milliseconds();
    }
    double gpu_cpu_ms = std::accumulate(times.begin(), times.end(), 0.0) / 10;
    double gpu_cpu_bw = effective_bandwidth_gbps(bytes * 2, gpu_cpu_ms);

    cpu_running = false;
    cpu_thread.join();

    printf("  GPU + CPU:    time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
           gpu_cpu_ms, gpu_cpu_bw, 100.0 * gpu_cpu_bw / GB10_PEAK_BW_GBPS);
    printf("  Bandwidth loss: %.1f%%  (UMA contention)\n",
           100.0 * (1.0 - gpu_cpu_bw / gpu_only_bw));

    free(h_hammer);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_header("CUDA Memory Bandwidth Lab — GB10 DGX Spark");
    printf("  Peak bandwidth: %.1f GB/s (LPDDR5X-8533, 256-bit bus)\n", GB10_PEAK_BW_GBPS);
    printf("  Memory: 128 GB unified (CPU + GPU share LPDDR5X)\n");

    // Part B-C: Working-set sweep
    sweep_working_set();

    // Part C2: Stride sweep
    sweep_stride();

    // Part C3: Block size sweep
    sweep_block_size();

    // Part D: Coalesced vs non-coalesced
    compare_coalesced();

    // Part E: Random vs sequential
    compare_random();

    // Part F: Atomics
    test_atomic();

    // Part G: Allocation types
    compare_alloc_types();

    // Part H: UMA contention
    uma_contention();

    // Summary
    print_header("Summary");
    printf("  Peak bandwidth:    %.1f GB/s\n", GB10_PEAK_BW_GBPS);
    printf("  Expected sustained: ~180 GB/s reads, ~116 GB/s writes\n");
    printf("  Your results above show actual sustained bandwidth.\n");
    printf("  Key questions to answer from this data:\n");
    printf("    1. At what working-set size does bandwidth plateau?\n");
    printf("    2. What stride causes bandwidth to drop by 50%%?\n");
    printf("    3. How much bandwidth is lost to CPU contention on UMA?\n");
    printf("    4. What is your measured sustained bandwidth vs 273 GB/s peak?\n\n");
    printf("  Run 'make profile' to capture Nsight Compute metrics.\n");

    return 0;
}
