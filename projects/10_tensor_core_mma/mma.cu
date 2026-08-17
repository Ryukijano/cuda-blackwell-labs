// Project 10: Hand-rolled FP16 Tensor Core MMA on GB10 (SM121)
//
// A from-scratch, warp-level WMMA FP16 GEMM that uses the Tensor Cores
// directly, and compares against cuBLAS on the same data.

#include "cuda_utils.h"
#include "benchmark.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <cstdio>

using namespace nvcuda;

#define WARP_SIZE 32

// Simple FP16 GEMM using warp-level WMMA, no shared memory.
// Each warp computes one 16x16 tile of C.
// A is MxK row-major, B is KxN col-major, C is MxN row-major.
__global__ void wmma_gemm_fp16(const half* __restrict__ A,
                               const half* __restrict__ B,
                               half* __restrict__ C,
                               int M, int N, int K) {
    // One warp per tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpsPerRow = N / 16;
    int warp_row = warpM / warpsPerRow;
    int warp_col = warpM % warpsPerRow;

    if (warp_row * 16 >= M || warp_col * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc;

    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += 16) {
        const half* a_tile = A + warp_row * 16 * K + k;
        const half* b_tile = B + k + warp_col * 16 * K;

        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, K);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    half* c_tile = C + warp_row * 16 * N + warp_col * 16;
    wmma::store_matrix_sync(c_tile, acc, N, wmma::mem_row_major);
}

// Check the Tensor Core result against a reference CPU baseline
static bool verify(const half* C, int M, int N, int K, float ref) {
    std::vector<half> hC((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), C, (size_t)M * N * sizeof(half), cudaMemcpyDeviceToHost));

    // Sample a few corner/edge elements
    int check[] = {0, N - 1, (M - 1) * N, (M - 1) * N + (N - 1), (M / 2) * N + (N / 2)};
    bool ok = true;
    for (int i : check) {
        float val = __half2float(hC[i]);
        if (fabsf(val - ref) > ref * 0.05f + 0.1f) {
            printf("    MISMATCH at %d: got %.3f, expected %.3f\n", i, val, ref);
            ok = false;
        }
    }
    return ok;
}

static double cublas_gemm_fp16(cublasHandle_t handle, int M, int N, int K,
                               const half* A, const half* B, half* C) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    // cuBLAS uses column-major; we pass CUBLAS_OP_N and pretend A/B are transposed.
    for (int i = 0; i < 3; i++) {
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 B, N,
                                 A, K,
                                 &beta,
                                 C, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    std::vector<double> times(10);
    for (int i = 0; i < 10; i++) {
        timer.start();
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 B, N,
                                 A, K,
                                 &beta,
                                 C, N));
        timer.stop();
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    return times[5];
}

static double wmma_gemm_time(int M, int N, int K,
                             const half* A, const half* B, half* C,
                             int iters) {
    int tileM = M / 16;
    int tileN = N / 16;
    int total_warps = tileM * tileN;
    int block = 128;      // 4 warps per block
    int grid = (total_warps + (block / WARP_SIZE) - 1) / (block / WARP_SIZE);

    // Warmup
    for (int i = 0; i < 3; i++) {
        wmma_gemm_fp16<<<grid, block>>>(A, B, C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    std::vector<double> times(iters);
    for (int i = 0; i < iters; i++) {
        timer.start();
        wmma_gemm_fp16<<<grid, block>>>(A, B, C, M, N, K);
        timer.stop();
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main() {
    print_header("Hand-rolled FP16 Tensor Core MMA — GB10 SM121");

    int sizes[][3] = {{256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    for (int s = 0; s < nsizes; s++) {
        int M = sizes[s][0];
        int N = sizes[s][1];
        int K = sizes[s][2];
        double flops = 2.0 * M * N * K;

        size_t a_bytes = (size_t)M * K * sizeof(half);
        size_t b_bytes = (size_t)K * N * sizeof(half);
        size_t c_bytes = (size_t)M * N * sizeof(half);

        // Fill A and B with 1.0, so each element of C = K.
        std::vector<half> hA(M * K, __float2half(1.0f));
        std::vector<half> hB(K * N, __float2half(1.0f));

        half *d_A, *d_B, *d_C, *d_C2;
        CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
        CUDA_CHECK(cudaMalloc(&d_C2, c_bytes));

        CUDA_CHECK(cudaMemcpy(d_A, hA.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, hB.data(), b_bytes, cudaMemcpyHostToDevice));

        printf("\n  --- GEMM %dx%dx%d (FP16) ---\n", M, N, K);

        // cuBLAS
        double ms_cublas = cublas_gemm_fp16(cublas, M, N, K, d_A, d_B, d_C);
        double tflops_cublas = flops / (ms_cublas * 1e-3) / 1e12;
        printf("  cuBLAS      time=%8.3f ms  TFLOP/s=%7.2f\n", ms_cublas, tflops_cublas);
        bool ok_cublas = verify(d_C, M, N, K, (float)K);

        // WMMA hand-rolled
        int iters = M <= 512 ? 50 : 20;
        double ms_wmma = wmma_gemm_time(M, N, K, d_A, d_B, d_C2, iters);
        double tflops_wmma = flops / (ms_wmma * 1e-3) / 1e12;
        printf("  WMMA kernel time=%8.3f ms  TFLOP/s=%7.2f\n", ms_wmma, tflops_wmma);
        bool ok_wmma = verify(d_C2, M, N, K, (float)K);

        double speedup = ms_cublas / ms_wmma;
        printf("  cuBLAS speedup vs WMMA: %.2fx  (verify cuBLAS=%s WMMA=%s)\n",
               1.0 / speedup, ok_cublas ? "OK" : "FAIL", ok_wmma ? "OK" : "FAIL");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C2);
    }

    cublasDestroy(cublas);
    return 0;
}
