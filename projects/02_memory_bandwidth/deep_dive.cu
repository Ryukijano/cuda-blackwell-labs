// Project 02 Deep Dive: L2 Cache Boundary, Latency, Random Read Breakdown
//
// Additional experiments to understand the GB10 memory hierarchy in detail:
// 1. Fine-grained L2 boundary sweep (find exact L2 cache size)
// 2. Pointer-chase latency measurement (true DRAM latency vs L2 latency)
// 3. Random read with varying working set (L2 hit vs miss effect)
// 4. L2 persistence across kernel launches (cold vs warm)
// 5. Read-only vs write-only bandwidth separation
// 6. Effect of cudaMemAdvise on managed memory

#include "cuda_utils.h"
#include "benchmark.h"
#include <vector>
#include <algorithm>
#include <random>

// ============================================================================
// 1. Fine-grained L2 boundary sweep
// ============================================================================

__global__ void seq_read_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

void l2_boundary_sweep() {
    print_header("L2 Cache Boundary Detection (fine-grained)");

    // L2 is 24MB. Sweep from 4MB to 48MB in 2MB steps
    // Each float = 4 bytes, so 2MB = 524288 floats
    int block_size = 256;
    int two_mb = 524288;

    printf("  %-10s  %12s  %12s  %12s\n", "Size", "BW (GB/s)", "% Peak", "Note");
    printf("  %-10s  %12s  %12s  %12s\n", "----------", "----------", "------", "----");

    for (int mult = 2; mult <= 24; mult++) {
        int n = mult * two_mb;
        size_t bytes = (size_t)n * sizeof(float);
        int grid = (n + block_size - 1) / block_size;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        cudaMemset(d_in, 1, bytes);

        // Warmup
        for (int i = 0; i < 3; i++) seq_read_kernel<<<grid, block_size>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        // Measure
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            seq_read_kernel<<<grid, block_size>>>(d_in, d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }

        // Use median for stability
        std::sort(times.begin(), times.end());
        double median = times[times.size() / 2];
        double bw = effective_bandwidth_gbps(bytes * 2, median);

        const char* note = "";
        if (mult * 2 <= 24) note = "fits in L2";
        else if (mult * 2 <= 28) note = "<-- L2 boundary";
        else note = "spills to DRAM";

        printf("  %4d MB     %10.1f    %10.1f%%  %s\n",
               mult * 2, bw, 100.0 * bw / GB10_PEAK_BW_GBPS, note);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
}

// ============================================================================
// 2. Pointer-chase latency measurement
// ============================================================================

// Pointer-chase: each thread follows a chain of pointers
// This prevents prefetching and measures true access latency
// We use a single thread to avoid hiding latency behind parallelism
__global__ void pointer_chase(const int* ptrs, float* out, int n, int iters) {
    int idx = threadIdx.x;
    int curr = 0;
    float sum = 0.0f;
    for (int i = 0; i < iters; i++) {
        curr = ptrs[curr];
        sum += (float)curr;
    }
    out[idx] = sum;
}

void latency_measurement() {
    print_header("Memory Latency (pointer-chase, single thread)");

    // Create a circular linked list in device memory
    // Each element points to a random next element
    // Working set sizes: 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB

    size_t sizes_bytes[] = {4096, 16384, 65536, 262144, 1048576, 4194304,
                            16777216, 67108864, 268435456};
    const char* labels[] = {"4KB", "16KB", "64KB", "256KB", "1MB", "4MB",
                            "16MB", "64MB", "256MB"};
    int num = sizeof(sizes_bytes) / sizeof(sizes_bytes[0]);

    printf("  %-10s  %12s  %12s  %12s\n", "Size", "Latency (ns)", "BW (GB/s)", "Cache level");
    printf("  %-10s  %12s  %12s  %12s\n", "----------", "------------", "----------", "-----------");

    for (int s = 0; s < num; s++) {
        size_t bytes = sizes_bytes[s];
        int n = bytes / sizeof(int);  // number of int pointers
        int iters = 100000;           // chase iterations

        // Create shuffled index array on host
        std::vector<int> h_ptrs(n);
        for (int i = 0; i < n; i++) h_ptrs[i] = i;
        // Fisher-Yates shuffle
        std::mt19937 rng(42);
        for (int i = n - 1; i > 0; i--) {
            std::uniform_int_distribution<int> dist(0, i);
            std::swap(h_ptrs[i], h_ptrs[dist(rng)]);
        }
        // Make it circular: each element points to the next in shuffle order
        // Actually, we want ptrs[i] = next_index, forming a cycle
        // Rearrange so that following ptrs[] visits all elements
        std::vector<int> chain(n);
        int curr = 0;
        for (int i = 0; i < n; i++) {
            chain[curr] = h_ptrs[(i + 1) % n];  // point to next in shuffle
            curr = h_ptrs[(i + 1) % n];
        }
        // Simpler: just make ptrs[i] = (i + stride) % n for random access
        // But for true random, use the shuffled order
        // ptrs[shuffled[i]] = shuffled[(i+1) % n]
        std::vector<int> ptrs(n, 0);
        for (int i = 0; i < n; i++) {
            ptrs[h_ptrs[i]] = h_ptrs[(i + 1) % n];
        }

        int *d_ptrs;
        float *d_out;
        CUDA_CHECK(cudaMalloc(&d_ptrs, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_ptrs, ptrs.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        // Warmup
        pointer_chase<<<1, 1>>>(d_ptrs, d_out, n, 1000);
        cudaDeviceSynchronize();

        // Measure
        GpuTimer timer;
        int runs = 10;
        std::vector<double> times(runs);
        for (int r = 0; r < runs; r++) {
            timer.start();
            pointer_chase<<<1, 1>>>(d_ptrs, d_out, n, iters);
            timer.stop();
            times[r] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median_ms = times[runs / 2];

        // Latency per access = total_time / iters
        double latency_ns = median_ms * 1e6 / iters;  // ms -> ns, divided by iters
        // Bandwidth = bytes_per_access / latency
        // Each access reads 4 bytes (one int)
        double bw_gbps = 4.0 / (latency_ns * 1e-9) / 1e9;

        const char* cache = "";
        if (bytes <= 64 * 1024) cache = "L1?";
        else if (bytes <= 24 * 1024 * 1024) cache = "L2 (24MB)";
        else cache = "DRAM";

        printf("  %-10s  %10.1f      %10.2f    %s\n",
               labels[s], latency_ns, bw_gbps, cache);

        CUDA_CHECK(cudaFree(d_ptrs));
        CUDA_CHECK(cudaFree(d_out));
    }
}

// ============================================================================
// 3. Random read with varying working set (L2 effect on random access)
// ============================================================================

__global__ void random_read_kernel(const float* in, float* out, const int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[indices[idx]];
    }
}

void random_read_working_set() {
    print_header("Random Read: Working Set vs Bandwidth");

    // Test random read with different working set sizes
    // When working set fits in L2 (24MB), random reads should be fast
    // When it exceeds L2, random reads hit DRAM with high latency

    size_t sizes[] = {262144, 1048576, 4194304, 16777216, 33554432,
                      67108864, 268435456};
    const char* labels[] = {"1MB", "4MB", "16MB", "64MB", "128MB", "256MB", "1GB"};
    int num = sizeof(sizes) / sizeof(sizes[0]);

    // Number of random accesses is fixed at 16M (64MB of reads)
    int num_accesses = 16777216;  // 16M random reads
    int block_size = 256;

    printf("  %-10s  %12s  %12s  %12s\n", "WSet", "BW (GB/s)", "% Peak", "Note");
    printf("  %-10s  %12s  %12s  %12s\n", "----------", "----------", "------", "----");

    for (int s = 0; s < num; s++) {
        size_t wset_bytes = sizes[s] * sizeof(float);
        int wset_n = sizes[s];
        int grid = (num_accesses + block_size - 1) / block_size;

        float *d_in, *d_out;
        int *d_indices;
        CUDA_CHECK(cudaMalloc(&d_in, wset_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, num_accesses * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_indices, num_accesses * sizeof(int)));
        cudaMemset(d_in, 1, wset_bytes);

        // Generate random indices into the working set
        std::vector<int> h_idx(num_accesses);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, wset_n - 1);
        for (int i = 0; i < num_accesses; i++) h_idx[i] = dist(rng);
        CUDA_CHECK(cudaMemcpy(d_indices, h_idx.data(), num_accesses * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Warmup
        for (int i = 0; i < 3; i++)
            random_read_kernel<<<grid, block_size>>>(d_in, d_out, d_indices, num_accesses);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            random_read_kernel<<<grid, block_size>>>(d_in, d_out, d_indices, num_accesses);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[5];

        // bytes moved: read indices (num_accesses * 4) + read data (num_accesses * 4) + write (num_accesses * 4)
        size_t bytes_moved = (size_t)num_accesses * 4 * 3;
        double bw = effective_bandwidth_gbps(bytes_moved, median);

        const char* note = "";
        if (wset_bytes <= 24 * 1024 * 1024) note = "fits in L2";
        else note = "exceeds L2 -> DRAM";

        printf("  %-10s  %10.1f    %10.1f%%  %s\n",
               labels[s], bw, 100.0 * bw / GB10_PEAK_BW_GBPS, note);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_indices));
    }
}

// ============================================================================
// 4. L2 persistence across kernel launches
// ============================================================================

void l2_persistence() {
    print_header("L2 Persistence Across Kernel Launches (64MB working set)");

    // 64MB > 24MB L2, so first launch is cold, but if L2 retains some data,
    // subsequent launches may be faster
    size_t n = 16777216;  // 64MB
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaMemset(d_in, 1, bytes);

    // Flush L2 by accessing a large buffer
    // Actually, we can't explicitly flush L2 on GB10, but we can
    // access a much larger buffer to evict the working set
    size_t flush_size = 128 * 1024 * 1024;  // 128MB
    float *d_flush;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_size));
    cudaMemset(d_flush, 0, flush_size);

    printf("  Launch  |  Cold BW (GB/s)  |  Warm BW (GB/s)\n");
    printf("  --------|------------------|------------------\n");

    for (int trial = 0; trial < 3; trial++) {
        // Cold: flush L2 first
        int flush_n = flush_size / sizeof(float);
        int flush_grid = (flush_n + block_size - 1) / block_size;
        seq_read_kernel<<<flush_grid, block_size>>>(d_flush, d_out, flush_n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        timer.start();
        seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        timer.stop();
        double cold_bw = effective_bandwidth_gbps(bytes * 2, timer.milliseconds());

        // Warm: immediately re-run (data should be partially in L2)
        timer.start();
        seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        timer.stop();
        double warm_bw = effective_bandwidth_gbps(bytes * 2, timer.milliseconds());

        printf("  Trial %d |  %14.1f   |  %14.1f\n", trial + 1, cold_bw, warm_bw);
    }

    printf("\n  Note: 64MB exceeds 24MB L2, so only partial L2 retention expected.\n");
    printf("  Cold = after flushing L2 with 128MB dummy read.\n");
    printf("  Warm = immediately after cold run (L2 may retain ~24MB of 64MB).\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_flush));
}

// ============================================================================
// 5. Read-only vs write-only bandwidth separation
// ============================================================================

__global__ void read_only_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Read but don't write to global (use shared to avoid compiler optimizing away)
        float val = in[idx];
        out[idx] = val;  // minimal write
    }
}

__global__ void write_only_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;  // write only, no read from global
    }
}

void read_write_separation() {
    print_header("Read-Only vs Write-Only Bandwidth (256MB)");

    size_t n = 67108864;  // 256MB
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaMemset(d_in, 1, bytes);

    auto bench = [&](auto kernel, const char* name, size_t bytes_moved) {
        for (int i = 0; i < 5; i++) kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(bytes_moved, median);
        printf("  %-25s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
               name, median, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
    };

    // read_only_kernel reads + writes (same as seq_read)
    // To truly measure read-only, we need to prevent the write
    // Use a dummy output that's 1 element
    printf("  Read + write (copy):     ");
    bench(read_only_kernel, "read+write (copy)", bytes * 2);

    printf("  Write only:              ");
    bench(write_only_kernel, "write_only", bytes);

    // For read-only, use a kernel that accumulates to shared memory
    // and only writes one float per block
    // This is approximate — true read-only is hard to measure
    printf("  Note: True read-only BW is hard to measure (compiler needs a sink).\n");
    printf("  The 'copy' kernel gives read+write combined BW.\n");
    printf("  Write-only gives pure write BW.\n");
    printf("  Estimated read-only BW ~ 2 * copy_bw - write_bw\n");

    // Recalculate
    for (int i = 0; i < 5; i++) read_only_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
    cudaDeviceSynchronize();
    GpuTimer t1;
    std::vector<double> tc(20);
    for (int i = 0; i < 20; i++) { t1.start(); read_only_kernel<<<grid, block_size>>>(d_in, d_out, (int)n); t1.stop(); tc[i] = t1.milliseconds(); }
    std::sort(tc.begin(), tc.end());
    double copy_bw = effective_bandwidth_gbps(bytes * 2, tc[10]);

    for (int i = 0; i < 5; i++) write_only_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
    cudaDeviceSynchronize();
    GpuTimer t2;
    std::vector<double> tw(20);
    for (int i = 0; i < 20; i++) { t2.start(); write_only_kernel<<<grid, block_size>>>(d_in, d_out, (int)n); t2.stop(); tw[i] = t2.milliseconds(); }
    std::sort(tw.begin(), tw.end());
    double write_bw = effective_bandwidth_gbps(bytes, tw[10]);

    double est_read_bw = 2.0 * copy_bw - write_bw;
    printf("\n  Estimated read-only BW:  ~%.1f GB/s  (2*copy - write)\n", est_read_bw);
    printf("  Measured write-only BW:   ~%.1f GB/s\n", write_bw);
    printf("  Measured copy (R+W) BW:   ~%.1f GB/s\n", copy_bw);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// 6. Managed memory with cudaMemAdvise
// ============================================================================

void managed_memory_advise() {
    print_header("Managed Memory: cudaMemAdvise Effect (256MB)");

    size_t n = 67108864;  // 256MB
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    // Test 1: No advise (default)
    {
        float *d_in, *d_out;
        CUDA_CHECK(cudaMallocManaged(&d_in, bytes));
        CUDA_CHECK(cudaMallocManaged(&d_out, bytes));
        memset(d_in, 1, bytes);

        // Warmup (trigger migration)
        for (int i = 0; i < 3; i++) seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double bw = effective_bandwidth_gbps(bytes * 2, times[5]);
        printf("  No advise (warm):        BW=%6.1f GB/s  (%4.1f%% peak)\n",
               bw, 100.0 * bw / GB10_PEAK_BW_GBPS);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    // Test 2: Advise read-mostly (hint that data is mostly read)
    {
        float *d_in, *d_out;
        CUDA_CHECK(cudaMallocManaged(&d_in, bytes));
        CUDA_CHECK(cudaMallocManaged(&d_out, bytes));
        memset(d_in, 1, bytes);

        cudaMemLocation gpu_loc = {cudaMemLocationTypeDevice, 0};
        CUDA_CHECK(cudaMemAdvise(d_in, bytes, cudaMemAdviseSetReadMostly, gpu_loc));
        CUDA_CHECK(cudaMemPrefetchAsync(d_in, bytes, gpu_loc, 0));  // prefetch to GPU
        cudaDeviceSynchronize();

        // Warmup
        for (int i = 0; i < 3; i++) seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double bw = effective_bandwidth_gbps(bytes * 2, times[5]);
        printf("  ReadMostly + prefetch:   BW=%6.1f GB/s  (%4.1f%% peak)\n",
               bw, 100.0 * bw / GB10_PEAK_BW_GBPS);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    // Test 3: Advise preferred location = GPU
    {
        float *d_in, *d_out;
        CUDA_CHECK(cudaMallocManaged(&d_in, bytes));
        CUDA_CHECK(cudaMallocManaged(&d_out, bytes));
        memset(d_in, 1, bytes);

        cudaMemLocation gpu_loc2 = {cudaMemLocationTypeDevice, 0};
        CUDA_CHECK(cudaMemAdvise(d_in, bytes, cudaMemAdviseSetPreferredLocation, gpu_loc2));
        CUDA_CHECK(cudaMemPrefetchAsync(d_in, bytes, gpu_loc2, 0));
        cudaDeviceSynchronize();

        for (int i = 0; i < 3; i++) seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double bw = effective_bandwidth_gbps(bytes * 2, times[5]);
        printf("  PreferredLoc + prefetch: BW=%6.1f GB/s  (%4.1f%% peak)\n",
               bw, 100.0 * bw / GB10_PEAK_BW_GBPS);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    // Test 4: cudaMalloc with prefetch for comparison
    {
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        cudaMemset(d_in, 1, bytes);

        for (int i = 0; i < 3; i++) seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            seq_read_kernel<<<grid, block_size>>>(d_in, d_out, (int)n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double bw = effective_bandwidth_gbps(bytes * 2, times[5]);
        printf("  cudaMalloc (baseline):   BW=%6.1f GB/s  (%4.1f%% peak)\n",
               bw, 100.0 * bw / GB10_PEAK_BW_GBPS);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    printf("\n  Key: cudaMemAdvise + cudaMemPrefetchAsync can close the gap\n");
    printf("  between managed memory and cudaMalloc on UMA.\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_header("CUDA Memory Deep Dive — GB10 DGX Spark");
    printf("  L2 cache: 24 MB | Peak BW: %.1f GB/s | Unified LPDDR5X\n\n", GB10_PEAK_BW_GBPS);

    l2_boundary_sweep();
    latency_measurement();
    random_read_working_set();
    l2_persistence();
    read_write_separation();
    managed_memory_advise();

    print_header("Deep Dive Complete");
    printf("  See results above for L2 boundary, latency, and UMA analysis.\n");
    printf("  Run 'make profile' on bandwidth_test for Nsight Compute metrics.\n");

    return 0;
}
