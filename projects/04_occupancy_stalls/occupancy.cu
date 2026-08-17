// Project 04: Occupancy & Stall Experiments
// Phase 2 — Compiler and SM Literacy
//
// Systematically vary register pressure, shared memory, block size, and divergence
// to understand how the SM schedules warps on GB10 (SM121).
//
// GB10 SM121 specs (from Blackwell Tuning Guide + probe):
//   Max warps/SM:        48   (1536 threads)
//   Register file/SM:    65536 (64K x 32-bit)
//   Max registers/thread: 255
//   Max blocks/SM:       24   (from probe, not 32)
//   Shared mem/SM:       100 KB (configured)
//   Shared mem/block:    99 KB (opt-in)
//
// Build:  make
// Run:    make run
// Profile: make profile

#include "cuda_utils.h"
#include "benchmark.h"
#include <vector>
#include <algorithm>

// ============================================================================
// GB10 SM121 occupancy constants (from probe + Blackwell Tuning Guide)
// ============================================================================

static const int MAX_WARPS_PER_SM    = 48;
static const int MAX_THREADS_PER_SM  = 1536;
static const int MAX_BLOCKS_PER_SM   = 24;
static const int REGISTERS_PER_SM    = 65536;
static const int SHARED_MEM_PER_SM   = 100 * 1024;  // 100 KB configured
static const int SHARED_MEM_PER_BLOCK_MAX = 99 * 1024;  // 99 KB

// Calculate theoretical occupancy
int calc_occupancy(int threads_per_block, int regs_per_thread, int smem_per_block_bytes) {
    int warps_per_block = threads_per_block / 32;
    if (warps_per_block == 0) warps_per_block = 1;

    // Limit 1: registers
    int regs_per_block = regs_per_thread * threads_per_block;
    // Round up to allocation granularity (256 on Blackwell)
    regs_per_block = ((regs_per_block + 255) / 256) * 256;
    int blocks_by_regs = REGISTERS_PER_SM / regs_per_block;

    // Limit 2: shared memory
    int smem_per_block = ((smem_per_block_bytes + 127) / 128) * 128;  // 128-byte alignment
    int blocks_by_smem = smem_per_block > 0 ? SHARED_MEM_PER_SM / smem_per_block : MAX_BLOCKS_PER_SM;

    // Limit 3: max blocks per SM
    int blocks_by_limit = MAX_BLOCKS_PER_SM;

    // Limit 4: max warps per SM
    int blocks_by_warps = MAX_WARPS_PER_SM / warps_per_block;

    int active_blocks = std::min({blocks_by_regs, blocks_by_smem, blocks_by_limit, blocks_by_warps});
    int active_warps = active_blocks * warps_per_block;
    int occupancy_pct = 100 * active_warps / MAX_WARPS_PER_SM;

    return occupancy_pct;
}

// ============================================================================
// Part A: Register Pressure Experiment
// ============================================================================

template<int N_REGS>
__global__ void register_pressure(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float regs[N_REGS];
    #pragma unroll
    for (int j = 0; j < N_REGS; j++) regs[j] = (float)(i + j);

    // Prevent optimization: compute dependent sum
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < N_REGS; j++) sum += regs[j] * 0.001f;

    out[i] = sum;
}

// Helper to get register count from cuobjdump (we'll hardcode known values)
struct RegResult {
    int n_regs;
    int regs_per_thread;
    int occupancy;
    double time_ms;
    double bw_gbps;
};

void register_pressure_test(float* d_out, int n) {
    print_header("Register Pressure Experiment (block_size=256)");

    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    // We'll test different register pressures and query actual register usage
    // Using cudaFuncGetAttributes to get register count at runtime

    struct { int n_regs; void* func; const char* name; } kernels[] = {
        {8,   (void*)register_pressure<8>,   "reg_8"},
        {16,  (void*)register_pressure<16>,  "reg_16"},
        {32,  (void*)register_pressure<32>,  "reg_32"},
        {48,  (void*)register_pressure<48>,  "reg_48"},
        {64,  (void*)register_pressure<64>,  "reg_64"},
        {96,  (void*)register_pressure<96>,  "reg_96"},
        {128, (void*)register_pressure<128>, "reg_128"},
        {160, (void*)register_pressure<160>, "reg_160"},
        {192, (void*)register_pressure<192>, "reg_192"},
        {255, (void*)register_pressure<255>, "reg_255"},
    };

    printf("  %-10s  %12s  %12s  %12s  %12s  %12s\n",
           "N_REGS", "ActualRegs", "TheorOcc%%", "Time(ms)", "BW(GB/s)", "Limit");
    printf("  %-10s  %12s  %12s  %12s  %12s  %12s\n",
           "----------", "----------", "----------", "----------", "----------", "----------");

    for (auto& k : kernels) {
        // Get actual register count
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k.func));
        int actual_regs = attr.numRegs;

        // Calculate theoretical occupancy
        int occ = calc_occupancy(block_size, actual_regs, 0);

        // Determine limiting factor
        int warps_per_block = block_size / 32;
        int regs_per_block = actual_regs * block_size;
        regs_per_block = ((regs_per_block + 255) / 256) * 256;
        int blocks_by_regs = REGISTERS_PER_SM / regs_per_block;
        int blocks_by_warps = MAX_WARPS_PER_SM / warps_per_block;
        const char* limit = "";
        if (blocks_by_regs <= blocks_by_warps && blocks_by_regs <= MAX_BLOCKS_PER_SM)
            limit = "registers";
        else if (blocks_by_warps <= MAX_BLOCKS_PER_SM)
            limit = "warps";
        else
            limit = "blocks";

        // Warmup
        for (int i = 0; i < 3; i++) {
            if (k.n_regs == 8)   register_pressure<8><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 16)  register_pressure<16><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 32)  register_pressure<32><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 48)  register_pressure<48><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 64)  register_pressure<64><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 96)  register_pressure<96><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 128) register_pressure<128><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 160) register_pressure<160><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 192) register_pressure<192><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 255) register_pressure<255><<<grid, block_size>>>(d_out, n);
        }
        cudaDeviceSynchronize();

        // Benchmark
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            if (k.n_regs == 8)   register_pressure<8><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 16)  register_pressure<16><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 32)  register_pressure<32><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 48)  register_pressure<48><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 64)  register_pressure<64><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 96)  register_pressure<96><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 128) register_pressure<128><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 160) register_pressure<160><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 192) register_pressure<192><<<grid, block_size>>>(d_out, n);
            else if (k.n_regs == 255) register_pressure<255><<<grid, block_size>>>(d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);  // write only

        printf("  %-10d  %12d  %12d  %12.3f  %12.1f  %12s\n",
               k.n_regs, actual_regs, occ, median, bw, limit);
    }
}

// ============================================================================
// Part B: Shared Memory Experiment
// ============================================================================

// Use dynamic shared memory for flexibility (supports >48KB with opt-in)
__global__ void shared_mem_pressure_dyn(float* out, int n, int smem_floats) {
    extern __shared__ float smem[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread writes to and reads from shared memory
    smem[tid % smem_floats] = (float)i;
    __syncthreads();
    if (i < n) {
        out[i] = smem[tid % smem_floats];
    }
}

void shared_memory_test(float* d_out, int n) {
    print_header("Shared Memory Pressure Experiment (block_size=256)");

    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    // Set opt-in for dynamic shared memory up to 99KB
    cudaFuncSetAttribute(shared_mem_pressure_dyn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);

    int smem_kb_vals[] = {1, 4, 8, 16, 32, 48, 64, 96};
    int num = sizeof(smem_kb_vals) / sizeof(smem_kb_vals[0]);

    printf("  %-10s  %12s  %12s  %12s  %12s\n",
           "SMEM(KB)", "TheorOcc%%", "Time(ms)", "BW(GB/s)", "Limit");
    printf("  %-10s  %12s  %12s  %12s  %12s\n",
           "----------", "----------", "----------", "----------", "----------");

    for (int s = 0; s < num; s++) {
        int smem_kb = smem_kb_vals[s];
        int smem_floats = smem_kb * 1024 / sizeof(float);
        int smem_bytes = smem_floats * sizeof(float);

        // Calculate theoretical occupancy (assume ~32 regs)
        int occ = calc_occupancy(block_size, 32, smem_bytes);

        // Determine limiting factor
        int warps_per_block = block_size / 32;
        int smem_aligned = ((smem_bytes + 127) / 128) * 128;
        int blocks_by_smem = SHARED_MEM_PER_SM / smem_aligned;
        int blocks_by_warps = MAX_WARPS_PER_SM / warps_per_block;
        const char* limit = blocks_by_smem <= blocks_by_warps ? "shared mem" : "warps";

        // Warmup
        for (int i = 0; i < 3; i++)
            shared_mem_pressure_dyn<<<grid, block_size, smem_bytes>>>(d_out, n, smem_floats);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            shared_mem_pressure_dyn<<<grid, block_size, smem_bytes>>>(d_out, n, smem_floats);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);

        printf("  %-10d  %12d  %12.3f  %12.1f  %12s\n",
               smem_kb, occ, median, bw, limit);
    }
}

// ============================================================================
// Part C: Block Size Sweep (with register pressure)
// ============================================================================

void block_size_sweep(float* d_out, int n) {
    print_header("Block Size Sweep (register_pressure<64>)");

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num = sizeof(block_sizes) / sizeof(block_sizes[0]);

    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (void*)register_pressure<64>));
    int actual_regs = attr.numRegs;

    printf("  Register count for register_pressure<64>: %d\n\n", actual_regs);
    printf("  %-12s  %12s  %12s  %12s  %12s\n",
           "BlockSize", "TheorOcc%%", "Time(ms)", "BW(GB/s)", "ActiveBlocks");
    printf("  %-12s  %12s  %12s  %12s  %12s\n",
           "----------", "----------", "----------", "----------", "-----------");

    for (int b = 0; b < num; b++) {
        int bs = block_sizes[b];
        int grid = (n + bs - 1) / bs;
        int occ = calc_occupancy(bs, actual_regs, 0);

        int warps_per_block = bs / 32;
        int regs_per_block = actual_regs * bs;
        regs_per_block = ((regs_per_block + 255) / 256) * 256;
        int blocks_by_regs = REGISTERS_PER_SM / regs_per_block;
        int blocks_by_warps = MAX_WARPS_PER_SM / warps_per_block;
        int active_blocks = std::min({blocks_by_regs, blocks_by_warps, MAX_BLOCKS_PER_SM});

        // Warmup
        for (int i = 0; i < 3; i++)
            register_pressure<64><<<grid, bs>>>(d_out, n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            register_pressure<64><<<grid, bs>>>(d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);

        printf("  %-12d  %12d  %12.3f  %12.1f  %12d\n",
               bs, occ, median, bw, active_blocks);
    }
}

// ============================================================================
// Part D: Warp Divergence Experiment
// ============================================================================

__global__ void no_divergence(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = (float)i;
    // All threads take same path
    if (x >= 0.0f) {
        x = x * 2.0f + 1.0f;
    } else {
        x = x * 3.0f + 2.0f;
    }
    out[i] = x;
}

__global__ void half_divergence(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = (float)i;
    // Half threads take each branch (based on thread index)
    if (threadIdx.x % 2 == 0) {
        x = x * 2.0f + 1.0f;
    } else {
        x = x * 3.0f + 2.0f;
    }
    out[i] = x;
}

__global__ void interleaved_divergence(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = (float)i;
    // Every other thread diverges
    if (i % 2 == 0) {
        x = x * 2.0f + 1.0f;
    } else {
        x = x * 3.0f + 2.0f;
    }
    out[i] = x;
}

__global__ void full_divergence(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = (float)i;
    // Each thread takes a different number of iterations
    int iters = threadIdx.x % 32;
    for (int j = 0; j < iters; j++) {
        x = x * 1.001f + 0.001f;
    }
    out[i] = x;
}

void divergence_test(float* d_out, int n) {
    print_header("Warp Divergence Experiment (block_size=256)");

    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    struct { const char* name; void (*kernel)(float*, int); } kernels[] = {
        {"no_divergence",        no_divergence},
        {"half_divergence",      half_divergence},
        {"interleaved_divergence", interleaved_divergence},
        {"full_divergence",      full_divergence},
    };

    printf("  %-25s  %12s  %12s  %12s\n",
           "Kernel", "Time(ms)", "BW(GB/s)", "Note");
    printf("  %-25s  %12s  %12s  %12s\n",
           "-------------------------", "----------", "----------", "----");

    for (auto& k : kernels) {
        for (int i = 0; i < 3; i++) k.kernel<<<grid, block_size>>>(d_out, n);
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            k.kernel<<<grid, block_size>>>(d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);

        const char* note = "";
        if (std::string(k.name) == "no_divergence") note = "all same branch";
        else if (std::string(k.name) == "half_divergence") note = "even/odd split";
        else if (std::string(k.name) == "interleaved_divergence") note = "interleaved";
        else note = "per-thread loop count";

        printf("  %-25s  %12.3f  %12.1f  %12s\n",
               k.name, median, bw, note);
    }
}

// ============================================================================
// Part E: Instruction Dependency Experiment
// ============================================================================

__global__ void short_chain(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = 1.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) x = x * 1.001f + 0.001f;
    out[i] = x;
}

__global__ void long_chain(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = 1.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) x = x * 1.001f + 0.001f;
    out[i] = x;
}

__global__ void independent_ops(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0 = 1.0f, x1 = 2.0f, x2 = 3.0f, x3 = 4.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        x0 = x0 * 1.001f;
        x1 = x1 * 1.001f;
        x2 = x2 * 1.001f;
        x3 = x3 * 1.001f;
    }
    out[i] = x0 + x1 + x2 + x3;
}

__global__ void memory_bound(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i] * 2.0f + 1.0f;
}

void dependency_test(float* d_out, const float* d_in, int n) {
    print_header("Instruction Dependency Experiment (block_size=256)");

    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    printf("  %-25s  %12s  %12s  %12s\n",
           "Kernel", "Time(ms)", "BW(GB/s)", "Bottleneck");
    printf("  %-25s  %12s  %12s  %12s\n",
           "-------------------------", "----------", "----------", "-----------");

    // Short chain
    {
        for (int i = 0; i < 3; i++) short_chain<<<grid, block_size>>>(d_out, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            short_chain<<<grid, block_size>>>(d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);
        printf("  %-25s  %12.3f  %12.1f  %12s\n",
               "short_chain (4 deps)", median, bw, "compute (short)");
    }

    // Long chain
    {
        for (int i = 0; i < 3; i++) long_chain<<<grid, block_size>>>(d_out, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            long_chain<<<grid, block_size>>>(d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);
        printf("  %-25s  %12.3f  %12.1f  %12s\n",
               "long_chain (32 deps)", median, bw, "compute (long)");
    }

    // Independent ops
    {
        for (int i = 0; i < 3; i++) independent_ops<<<grid, block_size>>>(d_out, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            independent_ops<<<grid, block_size>>>(d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);
        printf("  %-25s  %12.3f  %12.1f  %12s\n",
               "independent_ops (4 ILP)", median, bw, "compute (ILP)");
    }

    // Memory bound
    {
        for (int i = 0; i < 3; i++) memory_bound<<<grid, block_size>>>(d_out, d_in, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            memory_bound<<<grid, block_size>>>(d_out, d_in, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float) * 2, median);
        printf("  %-25s  %12.3f  %12.1f  %12s\n",
               "memory_bound (copy+calc)", median, bw, "memory");
    }
}

// ============================================================================
// Part F: "High Occupancy ≠ Fast" Proof
// ============================================================================

// Case 1: High occupancy, memory-bound, slow
// 1024 threads/block → high occupancy, but each thread does random read → slow
__global__ void high_occ_slow(const float* in, float* out, const int* indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[indices[i]];  // random read = slow
}

// Case 2: Low occupancy, coalesced, fast
// 64 threads/block → low occupancy, but coalesced sequential read → fast
__global__ void low_occ_fast(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];  // coalesced = fast
}

// Case 3: High occupancy with register spill (force high reg count)
template<int N>
__global__ void high_occ_reg_spill(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float regs[N];
    #pragma unroll
    for (int j = 0; j < N; j++) regs[j] = (float)(i + j);
    float sum = 0;
    #pragma unroll
    for (int j = 0; j < N; j++) sum += regs[j];
    out[i] = sum;
}

void high_occ_not_fast(float* d_out, const float* d_in, int n) {
    print_header("High Occupancy ≠ Fast Proof");

    // Generate random indices
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, n * sizeof(int)));
    std::vector<int> h_idx(n);
    srand(42);
    for (int i = 0; i < n; i++) h_idx[i] = rand() % n;
    CUDA_CHECK(cudaMemcpy(d_indices, h_idx.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    printf("\n  Case 1: High occupancy (1024 threads) + random read = SLOW\n");
    {
        int bs = 1024;
        int grid = (n + bs - 1) / bs;
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, (void*)high_occ_slow));
        int occ = calc_occupancy(bs, attr.numRegs, 0);

        for (int i = 0; i < 3; i++) high_occ_slow<<<grid, bs>>>(d_in, d_out, d_indices, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            high_occ_slow<<<grid, bs>>>(d_in, d_out, d_indices, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float) * 2, median);
        printf("    Occupancy: %d%%, Time: %.3f ms, BW: %.1f GB/s (%.1f%% peak)\n",
               occ, median, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
    }

    printf("\n  Case 2: Low occupancy (64 threads) + coalesced read = FAST\n");
    {
        int bs = 64;
        int grid = (n + bs - 1) / bs;
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, (void*)low_occ_fast));
        int occ = calc_occupancy(bs, attr.numRegs, 0);

        for (int i = 0; i < 3; i++) low_occ_fast<<<grid, bs>>>(d_in, d_out, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            low_occ_fast<<<grid, bs>>>(d_in, d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float) * 2, median);
        printf("    Occupancy: %d%%, Time: %.3f ms, BW: %.1f GB/s (%.1f%% peak)\n",
               occ, median, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
    }

    printf("\n  Case 3: High occupancy (256 threads) + 255 registers = SPILLS\n");
    {
        int bs = 256;
        int grid = (n + bs - 1) / bs;
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, (void*)high_occ_reg_spill<255>));
        int occ = calc_occupancy(bs, attr.numRegs, 0);

        for (int i = 0; i < 3; i++) high_occ_reg_spill<255><<<grid, bs>>>(d_out, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            high_occ_reg_spill<255><<<grid, bs>>>(d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);
        printf("    Occupancy: %d%%, Regs: %d, Time: %.3f ms, BW: %.1f GB/s\n",
               occ, attr.numRegs, median, bw);
    }

    printf("\n  Case 4: Low occupancy (256 threads) + 32 registers = NO SPILL\n");
    {
        int bs = 256;
        int grid = (n + bs - 1) / bs;
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, (void*)high_occ_reg_spill<32>));
        int occ = calc_occupancy(bs, attr.numRegs, 0);

        for (int i = 0; i < 3; i++) high_occ_reg_spill<32><<<grid, bs>>>(d_out, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            high_occ_reg_spill<32><<<grid, bs>>>(d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(n * sizeof(float), median);
        printf("    Occupancy: %d%%, Regs: %d, Time: %.3f ms, BW: %.1f GB/s\n",
               occ, attr.numRegs, median, bw);
    }

    printf("\n  Conclusion:\n");
    printf("    - Case 1 vs 2: Random access kills performance regardless of occupancy\n");
    printf("    - Case 3 vs 4: Register spills hurt more than low occupancy\n");
    printf("    - High occupancy is necessary but NOT sufficient for performance\n");

    CUDA_CHECK(cudaFree(d_indices));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_header("Occupancy & Stall Experiments — GB10 SM121");

    printf("  SM121 limits:\n");
    printf("    Max warps/SM:     %d  (%d threads)\n", MAX_WARPS_PER_SM, MAX_THREADS_PER_SM);
    printf("    Max blocks/SM:    %d\n", MAX_BLOCKS_PER_SM);
    printf("    Registers/SM:     %d\n", REGISTERS_PER_SM);
    printf("    Shared mem/SM:    %d KB\n", SHARED_MEM_PER_SM / 1024);
    printf("    Shared mem/block: %d KB (opt-in)\n\n", SHARED_MEM_PER_BLOCK_MAX / 1024);

    int n = 1 << 22;  // 4M elements = 16MB (fits in L2 for consistent results)
    size_t bytes = n * sizeof(float);

    float *d_out, *d_in;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    cudaMemset(d_in, 0, bytes);

    // Part A: Register pressure
    register_pressure_test(d_out, n);

    // Part B: Shared memory
    shared_memory_test(d_out, n);

    // Part C: Block size sweep
    block_size_sweep(d_out, n);

    // Part D: Divergence
    divergence_test(d_out, n);

    // Part E: Instruction dependency
    dependency_test(d_out, d_in, n);

    // Part F: High occupancy ≠ fast
    high_occ_not_fast(d_out, d_in, n);

    // Summary
    print_header("Summary");
    printf("  Run 'make profile' to capture Nsight Compute stall metrics.\n");
    printf("  Key metrics to examine:\n");
    printf("    sm__warps_active.avg.per_cycle_active     — achieved occupancy\n");
    printf("    smsp__average_warps_issue_stalled_*       — stall reasons\n");
    printf("    smsp__inst_executed.avg.per_cycle_active  — IPC\n");
    printf("    smsp__warps_eligible.avg.per_cycle_active — eligible warps\n\n");

    printf("  Theoretical occupancy formula:\n");
    printf("    occ = min(regs_limit, smem_limit, blocks_limit, warps_limit)\n");
    printf("    regs_limit  = REGISTERS_PER_SM / (regs_per_thread * threads_per_block)\n");
    printf("    smem_limit  = SHARED_MEM_PER_SM / smem_per_block\n");
    printf("    blocks_limit = MAX_BLOCKS_PER_SM\n");
    printf("    warps_limit = MAX_WARPS_PER_SM / (threads_per_block / 32)\n");

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in));

    return 0;
}
