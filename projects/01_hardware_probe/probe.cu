// Project 01: GB10 Hardware Probe
// Phase 1 — Hardware and Memory Literacy
//
// Interrogates the DGX Spark's GB10 GPU and reports every relevant hardware
// property. Cross-checks CUDA-reported memory against Linux-reported memory
// to understand the unified memory architecture.
//
// Build:  make
// Run:    make run   (saves output to results/01_hardware_probe.txt)
// PTX:    make ptx   (generates probe.ptx for inspection)

#include "cuda_utils.h"

int main() {
    // ========================================================================
    // Part A: Device Properties
    // ========================================================================
    print_device_properties();

    // ========================================================================
    // Part B & C: Memory Cross-Check (UMA lesson)
    // ========================================================================
    print_memory_info();

    // ========================================================================
    // Part D: Compilation verification
    // ========================================================================
    print_header("Compilation Info");
    printf("  Compiled with: nvcc -arch=sm_121 -lineinfo -std=c++17\n");
    printf("  CUDA version:  %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    printf("  Driver version:  %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
    printf("  Runtime version: %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);

    // ========================================================================
    // Summary: What this means for CUDA programming on GB10
    // ========================================================================
    print_header("Key Takeaways for GB10 CUDA Programming");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("  1. This is a %s (SM %d.%d), NOT a datacenter B200.\n", prop.name, prop.major, prop.minor);
    printf("     Do NOT use TMEM, WGMMA, or DSMEM — these don't exist on SM121.\n");
    printf("     CUTLASS FP4 paths may produce silent garbage.\n\n");

    printf("  2. Memory is UNIFIED LPDDR5X, not HBM.\n");
    printf("     Peak bandwidth: ~273 GB/s (vs ~3000 GB/s on H100).\n");
    printf("     Bandwidth is your primary bottleneck. Optimize memory access\n");
    printf("     patterns before anything else.\n\n");

    printf("  3. %d SMs, ~%d CUDA cores (128/SM), ~%d Tensor Cores (5th gen).\n",
           prop.multiProcessorCount,
           prop.multiProcessorCount * 128,  // Blackwell: 128 CUDA cores per SM
           prop.multiProcessorCount * 8);   // approximate: 8 Tensor Cores per SM
    printf("     Max %d threads per SM, %d warps per SM.\n\n",
           prop.maxThreadsPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor / prop.warpSize);

    printf("  4. Shared memory: %lu KB per block (opt-in), %lu KB per SM.\n",
           prop.sharedMemPerBlockOptin / 1024,
           prop.sharedMemPerMultiprocessor / 1024);
    printf("     Use -arch=sm_121 and __launch_bounds__ to control occupancy.\n\n");

    printf("  5. Unified memory: CPU and GPU share the same physical LPDDR5X.\n");
    printf("     cudaMemGetInfo() underreports — always check /proc/meminfo.\n");
    printf("     Page cache competition can cause OOM in warm cache states.\n\n");

    printf("  6. Compile flags:\n");
    printf("     nvcc -arch=sm_121 -lineinfo -O2 kernel.cu -o kernel\n");
    printf("     -lineinfo is REQUIRED for ncu profiling\n");
    printf("     Use compute_121 for forward-compatible PTX\n\n");

    printf("============================================================\n");
    printf("  Project 01 complete. See results/01_hardware_probe.txt\n");
    printf("  for full output. Run 'make ptx' to generate probe.ptx.\n");
    printf("============================================================\n\n");

    return 0;
}
