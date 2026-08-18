// Project 11: Tensor Memory Accelerator (TMA) 2D Tile Copy
//
// Demonstrates the TMA engine on GB10 (SM121) by copying 2D tiles from
// global memory to shared memory with a hardware-managed tensor map, then
// writing the result back to global memory.

#include "cuda_utils.h"
#include <cuda.h>   // CUtensorMap, cuTensorMapEncodeTiled
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

#define TILE_H 64
#define TILE_W 64

// ---------------------------------------------------------------------------
// Inline PTX helpers (adapted from public learn-cuda reference patterns)
// ---------------------------------------------------------------------------
__device__ inline void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ inline void tma_2d_g2s(int dst, const void* tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr) : "memory");
}

__device__ inline void tma_2d_s2g(const void* tmap_ptr, int x, int y, int src) {
    // TMA store does not need an mbarrier for this simple benchmark; we just
    // commit/wait for the group to drain before measuring end time.
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];"
                 :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(src) : "memory");
}

__device__ inline void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ inline void cp_async_bulk_wait_group() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// ---------------------------------------------------------------------------
// Host-side tensor map construction
// ---------------------------------------------------------------------------
static void init_tmap_2d(CUtensorMap* tmap, const void* ptr,
                         int global_h, int global_w,
                         int tile_h, int tile_w,
                         CUtensorMapDataType dtype) {
    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank]       = {(uint64_t)global_w, (uint64_t)global_h};
    uint64_t globalStrides[rank-1] = {(uint64_t)global_w * 4};  // element size in bytes
    uint32_t boxDim[rank]          = {(uint32_t)tile_w, (uint32_t)tile_h};
    uint32_t elementStrides[rank]  = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        tmap,
        dtype,
        rank,
        (void*)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", (int)res);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// TMA 2D copy kernel: global -> shared (TMA) -> global (TMA store)
// ---------------------------------------------------------------------------
__global__ void tma_copy_2d(const __grid_constant__ CUtensorMap tmap_in,
                            const __grid_constant__ CUtensorMap tmap_out,
                            int H, int W) {
    __shared__ alignas(128) int smem[TILE_H * TILE_W];
    __shared__ alignas(8) uint64_t mbar;

    int mbar_addr = (int)__cvta_generic_to_shared(&mbar);

    if (threadIdx.x == 0) {
        mbarrier_init(mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    int x = blockIdx.x * TILE_W;
    int y = blockIdx.y * TILE_H;
    if (x >= W || y >= H) return;

    if (threadIdx.x == 0) {
        int smem_addr = (int)__cvta_generic_to_shared(smem);
        mbarrier_arrive_expect_tx(mbar_addr, TILE_H * TILE_W * sizeof(int));
        tma_2d_g2s(smem_addr, &tmap_in, x, y, mbar_addr);
    }

    // All threads wait for the TMA load to complete.
    mbarrier_wait(mbar_addr, 0);
    __syncthreads();

    // Optional in-kernel transform: add 1 to every element just to prove
    // the data really went through shared memory and is now resident.
    for (int i = threadIdx.x; i < TILE_H * TILE_W; i += blockDim.x) {
        smem[i] += 1;
    }
    __syncthreads();

    // Issue a TMA store back to global memory.
    if (threadIdx.x == 0) {
        int smem_addr = (int)__cvta_generic_to_shared(smem);
        tma_2d_s2g(&tmap_out, x, y, smem_addr);
        cp_async_bulk_commit_group();
    }

    // Wait for store group to drain before the kernel returns.
    cp_async_bulk_wait_group();
}

// ---------------------------------------------------------------------------
// Naive 2D tile copy for comparison: one int per thread.
// ---------------------------------------------------------------------------
__global__ void naive_copy_2d(const int* __restrict__ in, int* __restrict__ out,
                              int H, int W) {
    int block_x = blockIdx.x * TILE_W;
    int block_y = blockIdx.y * TILE_H;

    for (int i = threadIdx.x; i < TILE_H * TILE_W; i += blockDim.x) {
        int row = i / TILE_W;
        int col = i % TILE_W;
        int x = block_x + col;
        int y = block_y + row;
        if (x < W && y < H) {
            out[y * W + x] = in[y * W + x] + 1;
        }
    }
}

int main() {
    print_header("Tensor Memory Accelerator (TMA) 2D Tile Copy — GB10 SM121");

    int H = 4096, W = 4096;
    size_t N = (size_t)H * W;
    size_t bytes = N * sizeof(int);

    std::vector<int> h_in(N), h_out(N), h_ref(N);
    for (size_t i = 0; i < N; i++) h_in[i] = (int)i % 97;

    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    // Build TMA descriptors for input and output.
    CUtensorMap tmap_in, tmap_out;
    init_tmap_2d(&tmap_in, d_in, H, W, TILE_H, TILE_W, CU_TENSOR_MAP_DATA_TYPE_INT32);
    init_tmap_2d(&tmap_out, d_out, H, W, TILE_H, TILE_W, CU_TENSOR_MAP_DATA_TYPE_INT32);

    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);
    dim3 block(256);

    // Warmup
    for (int i = 0; i < 5; i++) tma_copy_2d<<<grid, block>>>(tmap_in, tmap_out, H, W);
    CUDA_CHECK(cudaDeviceSynchronize());

    // TMA copy benchmark
    GpuTimer tma_timer;
    std::vector<double> tma_times(51);
    for (int i = 0; i < 51; i++) {
        tma_timer.start();
        tma_copy_2d<<<grid, block>>>(tmap_in, tmap_out, H, W);
        tma_timer.stop();
        tma_times[i] = tma_timer.milliseconds();
    }
    std::sort(tma_times.begin(), tma_times.end());
    double tma_ms = tma_times[25];

    // Naive copy benchmark
    for (int i = 0; i < 5; i++) naive_copy_2d<<<grid, block>>>(d_in, d_out, H, W);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer naive_timer;
    std::vector<double> naive_times(51);
    for (int i = 0; i < 51; i++) {
        naive_timer.start();
        naive_copy_2d<<<grid, block>>>(d_in, d_out, H, W);
        naive_timer.stop();
        naive_times[i] = naive_timer.milliseconds();
    }
    std::sort(naive_times.begin(), naive_times.end());
    double naive_ms = naive_times[25];

    // Verify
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < N; i++) {
        int expected = h_in[i] + 1;
        if (h_out[i] != expected) {
            if (ok) printf("  MISMATCH at %zu: got %d, expected %d\n", i, h_out[i], expected);
            ok = false;
            break;
        }
    }

    double bytes_moved = 2.0 * bytes;  // read + write
    double tma_bw = effective_bandwidth_gbps(bytes_moved, tma_ms);
    double naive_bw = effective_bandwidth_gbps(bytes_moved, naive_ms);

    printf("\n  --- %dx%d int copy ---\n", H, W);
    printf("  TMA     time=%.4f ms  BW=%.1f GB/s  verify=%s\n", tma_ms, tma_bw, ok ? "OK" : "FAIL");
    printf("  Naive   time=%.4f ms  BW=%.1f GB/s\n", naive_ms, naive_bw);
    printf("  Speedup: %.2fx\n", naive_ms / tma_ms);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
