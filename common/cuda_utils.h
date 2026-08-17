#pragma once

// CUDA Blackwell Labs — Common Utilities
// Shared header for all projects. Provides error checking macros,
// timing utilities, and GB10-specific constants.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <chrono>
#include <string>

// ============================================================================
// GB10 DGX Spark Constants (SM121, compute capability 12.1)
// ============================================================================

// Updated from probe.cu results on this specific GB10:
#define GB10_SM_COUNT          48     // actual: 48 SMs (not 10 as docs suggest)
#define GB10_WARP_SIZE         32
#define GB10_MAX_THREADS_BLOCK 1024
#define GB10_MAX_THREADS_SM    1536
#define GB10_WARPS_PER_SM      48     // 1536 / 32
#define GB10_BLOCKS_PER_SM     24     // from cudaDevAttrMaxBlocksPerMultiprocessor
#define GB10_SHARED_MEM_BLOCK  99     // KB (opt-in per block)
#define GB10_SHARED_MEM_SM     100    // KB per SM
#define GB10_REGISTERS_PER_SM  65536
#define GB10_REGISTERS_PER_BLOCK 65536
#define GB10_L2_CACHE_KB      24576  // 24 MB (actual from probe)
#define GB10_MEM_CLOCK_MHZ    8533   // LPDDR5X data rate (MT/s)
#define GB10_MEM_BUS_WIDTH    256    // bits
// Peak bandwidth = data_rate * bus_width / 8 = 8533e6 * 256 / 8 = ~273 GB/s
// (No * 2 factor: cudaDevAttrMemoryClockRate returns data rate, not base clock)
#define GB10_PEAK_BW_GBPS     273.0

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error (last) at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t err = (call);                                           \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, (int)err);                             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// Timing Utilities
// ============================================================================

class CpuTimer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop()  { stop_  = std::chrono::high_resolution_clock::now(); }
    double milliseconds() const {
        return std::chrono::duration<double, std::milli>(stop_ - start_).count();
    }
    double seconds() const {
        return std::chrono::duration<double>(stop_ - start_).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_, stop_;
};

// GPU timer using CUDA events (more accurate than CPU timer for GPU work)
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }
    float milliseconds() const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    float seconds() const { return milliseconds() / 1000.0f; }
private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Bandwidth Calculation Helper
// ============================================================================

// effective_bandwidth_gbps(bytes, time_ms) -> GB/s
// bytes: total bytes transferred (read + written)
// time_ms: elapsed time in milliseconds
inline double effective_bandwidth_gbps(size_t bytes, double time_ms) {
    return (double)bytes / (time_ms * 1e-3) / 1e9;
}

// ============================================================================
// Pretty Printing
// ============================================================================

inline void print_header(const char* title) {
    printf("\n");
    printf("============================================================\n");
    printf("  %s\n", title);
    printf("============================================================\n\n");
}

inline void print_kv(const char* key, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("  %-30s ", key);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

inline void print_separator() {
    printf("  %-30s %s\n", "------------------------------", "----------");
}

// ============================================================================
// Memory Utilities (GB10 UMA-specific)
// ============================================================================

// Print memory info from both CUDA and Linux, highlighting UMA discrepancies
inline void print_memory_info() {
    print_header("Memory Cross-Check (GB10 Unified Memory)");

    size_t cuda_free, cuda_total;
    CUDA_CHECK(cudaMemGetInfo(&cuda_free, &cuda_total));

    printf("  CUDA-reported:\n");
    printf("    Free:    %.2f GB\n", cuda_free / 1e9);
    printf("    Total:   %.2f GB\n", cuda_total / 1e9);

    // Read /proc/meminfo
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        long mem_total = 0, mem_free = 0, mem_avail = 0, buffers = 0, cached = 0;
        long swap_total = 0, swap_free = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemTotal:", 9) == 0)      sscanf(line + 9, "%ld", &mem_total);
            else if (strncmp(line, "MemFree:", 8) == 0)   sscanf(line + 8, "%ld", &mem_free);
            else if (strncmp(line, "MemAvailable:", 13) == 0) sscanf(line + 13, "%ld", &mem_avail);
            else if (strncmp(line, "Buffers:", 8) == 0)   sscanf(line + 8, "%ld", &buffers);
            else if (strncmp(line, "Cached:", 7) == 0)    sscanf(line + 7, "%ld", &cached);
            else if (strncmp(line, "SwapTotal:", 10) == 0) sscanf(line + 10, "%ld", &swap_total);
            else if (strncmp(line, "SwapFree:", 9) == 0)  sscanf(line + 9, "%ld", &swap_free);
        }
        fclose(f);

        printf("\n  Linux-reported (/proc/meminfo):\n");
        printf("    MemTotal:     %.2f GB\n", mem_total / 1e6);
        printf("    MemFree:      %.2f GB\n", mem_free / 1e6);
        printf("    MemAvailable: %.2f GB\n", mem_avail / 1e6);
        printf("    Buffers:      %.2f GB\n", buffers / 1e6);
        printf("    Cached:       %.2f GB\n", cached / 1e6);
        printf("    SwapTotal:    %.2f GB\n", swap_total / 1e6);
        printf("    SwapFree:     %.2f GB\n", swap_free / 1e6);

        double page_cache = (buffers + cached) / 1e6;
        double theoretical = (mem_avail + page_cache + swap_free) / 1e6;
        printf("\n  Analysis:\n");
        printf("    Page cache (Buffers+Cached):  %.2f GB\n", page_cache);
        printf("    Theoretical allocatable:      %.2f GB  (avail + cache + swap)\n", theoretical);
        printf("    CUDA sees:                    %.2f GB\n", cuda_free / 1e9);
        printf("    Discrepancy:                  %.2f GB\n",
               theoretical - cuda_free / 1e9);
        printf("\n  NOTE: On GB10's unified LPDDR5X architecture, cudaMemGetInfo()\n");
        printf("  underreports because the OS can reclaim page cache and swap.\n");
        printf("  Always cross-check with /proc/meminfo before large allocations.\n");
    }
}

// ============================================================================
// Device Properties Printer
// ============================================================================

inline void print_device_properties() {
    print_header("GB10 Device Properties");

    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    printf("  Device count: %d\n", dev_count);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n  --- cudaGetDeviceProperties ---\n");
    printf("  %-30s %s\n",        "Name",                    prop.name);
    printf("  %-30s %d.%d\n",     "Compute capability",      prop.major, prop.minor);
    printf("  %-30s %d\n",        "Multi-processor count",   prop.multiProcessorCount);
    printf("  %-30s %d\n",        "Warp size",               prop.warpSize);
    printf("  %-30s %d\n",        "Max threads per block",   prop.maxThreadsPerBlock);
    printf("  %-30s %d x %d x %d\n", "Max threads dim",       prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  %-30s %d x %d x %d\n", "Max grid size",         prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  %-30s %lu KB\n",    "Shared mem per block",    prop.sharedMemPerBlock / 1024);
    printf("  %-30s %lu KB\n",    "Shared mem per SM",       prop.sharedMemPerMultiprocessor / 1024);
    printf("  %-30s %lu KB\n",    "Shared mem per block(opt)",prop.sharedMemPerBlockOptin / 1024);
    printf("  %-30s %lu KB\n",    "Total constant memory",   prop.totalConstMem / 1024);
    printf("  %-30s %d\n",        "Registers per block",     prop.regsPerBlock);
    printf("  %-30s %d\n",        "Registers per SM",        prop.regsPerMultiprocessor);
    printf("  %-30s %d KB\n",     "L2 cache size",           prop.l2CacheSize / 1024);
    printf("  %-30s %d bits\n",   "Memory bus width",        prop.memoryBusWidth);
    printf("  %-30s %d\n",        "Concurrent kernels",      prop.concurrentKernels);
    printf("  %-30s %d\n",        "Async engine count",      prop.asyncEngineCount);
    printf("  %-30s %s\n",        "Unified addressing",      prop.unifiedAddressing ? "Yes" : "No");
    printf("  %-30s %s\n",        "Managed memory",          prop.managedMemory ? "Yes" : "No");
    printf("  %-30s %s\n",        "Memory pools",            prop.memoryPoolsSupported ? "Yes" : "No");
    printf("  %-30s %s\n",        "Cooperative launch",      prop.cooperativeLaunch ? "Yes" : "No");
    printf("  %-30s %d\n",        "PCI bus ID",              prop.pciBusID);
    printf("  %-30s %s\n",        "TCC driver mode",         prop.tccDriver ? "Yes" : "No");

    // Additional attributes via cudaDeviceGetAttribute
    printf("\n  --- cudaDeviceGetAttribute (extended) ---\n");

    struct { const char* name; cudaDeviceAttr attr; bool is_bool; bool is_kb; } attrs[] = {
        {"Max blocks per SM",              cudaDevAttrMaxBlocksPerMultiprocessor, false, false},
        {"Max threads per SM",             cudaDevAttrMaxThreadsPerMultiProcessor, false, false},
        {"Warp size (attr)",               cudaDevAttrWarpSize, false, false},
        {"Max shared mem per block optin", cudaDevAttrMaxSharedMemoryPerBlockOptin, false, true},
        {"Max shared mem per SM",          cudaDevAttrMaxSharedMemoryPerMultiprocessor, false, true},
        {"L2 cache size (attr)",           cudaDevAttrL2CacheSize, false, true},
        {"Memory clock rate",              cudaDevAttrMemoryClockRate, false, false},
        {"Global L1 cache supported",      cudaDevAttrGlobalL1CacheSupported, true, false},
        {"Local L1 cache supported",       cudaDevAttrLocalL1CacheSupported, true, false},
        {"Persisting L2 cache max size",   cudaDevAttrMaxPersistingL2CacheSize, false, true},
        {"Multi-board GPU group ID",       cudaDevAttrMultiGpuBoardGroupID, false, false},
        {"Integrated (iGPU)",              cudaDevAttrIntegrated, true, false},
        {"Can map host memory",            cudaDevAttrCanMapHostMemory, true, false},
        {"Concurrent managed access",      cudaDevAttrConcurrentManagedAccess, true, false},
        {"Pageable memory access",         cudaDevAttrPageableMemoryAccess, true, false},
        {"Pageable memory uses host pages",cudaDevAttrPageableMemoryAccessUsesHostPageTables, true, false},
        {"Memory pools supported",         cudaDevAttrMemoryPoolsSupported, true, false},
        {"Deferred mapping CUDA array",    cudaDevAttrDeferredMappingCudaArraySupported, true, false},
    };

    for (auto& a : attrs) {
        int val = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&val, a.attr, 0));
        if (a.is_bool) {
            printf("  %-30s %s\n", a.name, val ? "Yes" : "No");
        } else if (a.is_kb) {
            printf("  %-30s %d KB\n", a.name, val / 1024);
        } else if (strstr(a.name, "clock rate")) {
            printf("  %-30s %d MHz\n", a.name, val / 1000);
        } else {
            printf("  %-30s %d\n", a.name, val);
        }
    }

    // Calculate peak bandwidth from attributes
    // cudaDevAttrMemoryClockRate returns data rate in kHz (for DDR, this is MT/s)
    // Peak BW = data_rate * bus_width / 8 (no extra * 2 for DDR)
    int mem_clock_khz = 0, bus_width = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_bw = (double)mem_clock_khz * 1000.0 * (bus_width / 8) / 1e9;
    printf("  %-30s %.1f GB/s\n", "Peak memory bandwidth (calc)", peak_bw);
}
