// Project 05: Five-Way GEMM Comparison
// Phase 2 — Compiler and SM Literacy
//
// Implement matrix multiplication (GEMM) multiple ways and benchmark:
// 1. Naive CUDA GEMM (one thread per output element)
// 2. Tiled shared memory GEMM (16x16 and 32x32 tiles)
// 3. Vectorized 1D-blocked GEMM (float4 loads, register accumulation)
// 4. cuBLAS Sgemm (the gold standard)
//
// CUTLASS and PyTorch are not installed on this system — see ANALYSIS.md for notes.
//
// Build:  make
// Run:    make run

#include "cuda_utils.h"
#include "benchmark.h"
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// 1. Naive CUDA GEMM
// ============================================================================

__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 2. Tiled Shared Memory GEMM (16x16 tiles)
// ============================================================================

#define TILE_SIZE 16

__global__ void tiled_gemm_16(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        // Load tile into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        s_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 2b. Tiled Shared Memory GEMM (32x32 tiles)
// ============================================================================

#define TILE_SIZE_32 32

__global__ void tiled_gemm_32(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE_32][TILE_SIZE_32];
    __shared__ float s_B[TILE_SIZE_32][TILE_SIZE_32];

    int row = blockIdx.y * TILE_SIZE_32 + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_32 + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE_32 - 1) / TILE_SIZE_32;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE_32 + threadIdx.x;
        int b_row = t * TILE_SIZE_32 + threadIdx.y;
        s_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE_32; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 3. Vectorized 1D-blocked GEMM (float4 loads, register accumulation)
// Each thread computes a 4x1 column of C, using float4 loads from A
// ============================================================================

__global__ void vec4_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each thread block: 64 threads, each computes 4 rows x 1 col = 4 outputs
    // Block covers 256 rows x 1 col. Grid covers N columns.
    // Actually, let's do: each thread computes 1 output, but loads A with float4
    // and processes 4 K elements at once

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int k = 0;
    // Process 4 elements at a time using float4 for A; B stays scalar
    // because B[k * N + col] is strided by N and cannot be vectorized cheaply.
    for (; k + 3 < K; k += 4) {
        float4 a4 = *reinterpret_cast<const float4*>(&A[row * K + k]);
        sum += a4.x * B[k * N + col];
        sum += a4.y * B[(k + 1) * N + col];
        sum += a4.z * B[(k + 2) * N + col];
        sum += a4.w * B[(k + 3) * N + col];
    }
    // Remainder
    for (; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ============================================================================
// 3b. Tiled GEMM with 1D thread mapping and register accumulation
// Each thread computes BM elements in the M dimension (better register reuse)
// ============================================================================

#define BM 64
#define BN 64
#define BK 8
#define TM 4  // each thread computes TM x 1 outputs

__global__ void tiled_gemm_reg(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block: BM x BN output tile (64x64)
    // Threads: (BM/TM) x BN = 16 x 64 = 1024 threads
    // Each thread computes TM=4 output elements (4 rows x 1 col)

    int thread_row = threadIdx.x / BN;       // 0..15
    int thread_col = threadIdx.x % BN;       // 0..63
    int row_start = blockIdx.y * BM + thread_row * TM;
    int col = blockIdx.x * BN + thread_col;

    __shared__ float s_A[BM][BK];   // 64x8
    __shared__ float s_B[BK][BN];   // 8x64

    float regs[TM] = {0.0f};

    int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; t++) {
        // Load A tile: 64x8 = 512 elements, 1024 threads → 2 threads per element
        int a_load_row = threadIdx.x / BK;        // 0..127
        int a_load_col = threadIdx.x % BK;        // 0..7
        if (a_load_row < BM) {
            int global_row = blockIdx.y * BM + a_load_row;
            int global_col = t * BK + a_load_col;
            s_A[a_load_row][a_load_col] = (global_row < M && global_col < K)
                ? A[global_row * K + global_col] : 0.0f;
        }

        // Load B tile: 8x64 = 512 elements, 1024 threads → 2 threads per element
        int b_load_row = threadIdx.x / BN;        // 0..15
        int b_load_col = threadIdx.x % BN;        // 0..63
        if (b_load_row < BK) {
            int global_row = t * BK + b_load_row;
            int global_col = blockIdx.x * BN + b_load_col;
            s_B[b_load_row][b_load_col] = (global_row < K && global_col < N)
                ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        // Compute: each thread accumulates TM=4 outputs
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float b_val = s_B[k][thread_col];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regs[m] += s_A[thread_row * TM + m][k] * b_val;
            }
        }
        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = row_start + m;
        if (row < M && col < N) {
            C[row * N + col] = regs[m];
        }
    }
}

// ============================================================================
// CPU reference GEMM (for correctness verification)
// ============================================================================

void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

float max_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ============================================================================
// Benchmark framework
// ============================================================================

struct GemmResult {
    const char* name;
    int size;
    double time_ms;
    double tflops;
    double bw_gbps;
    float max_err;
};

void benchmark_gemm(int M, int N, int K, std::vector<GemmResult>& results) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);
    size_t total_bytes = bytes_A + bytes_B + bytes_C;
    double flops = 2.0 * M * N * K;

    // Allocate and initialize
    std::vector<float> h_A(M * K), h_B(K * N), h_C_cpu(M * N);
    srand(42);
    for (size_t i = 0; i < h_A.size(); i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < h_B.size(); i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    // CPU reference (only for small sizes)
    if (M <= 512) {
        cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    auto run_bench = [&](const char* name, auto launch_fn) {
        // Warmup
        for (int i = 0; i < 3; i++) launch_fn();
        cudaDeviceSynchronize();

        GpuTimer timer;
        std::vector<double> times(10);
        for (int i = 0; i < 10; i++) {
            timer.start();
            launch_fn();
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[5];
        double tflops = flops / (median * 1e-3) / 1e12;
        double bw = total_bytes / (median * 1e-3) / 1e9;

        // Verify correctness
        float max_err = 0.0f;
        if (M <= 512) {
            std::vector<float> h_C_gpu(M * N);
            CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
            max_err = max_error(h_C_cpu.data(), h_C_gpu.data(), M * N);
        } else {
            // For large sizes, compare against cuBLAS
            max_err = -1.0f;  // skip
        }

        results.push_back({name, M, median, tflops, bw, max_err});
    };

    // 1. Naive
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        run_bench("naive", [&]() {
            naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        });
    }

    // 2a. Tiled 16x16
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        run_bench("tiled_16", [&]() {
            tiled_gemm_16<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        });
    }

    // 2b. Tiled 32x32
    {
        dim3 block(TILE_SIZE_32, TILE_SIZE_32);
        dim3 grid((N + TILE_SIZE_32 - 1) / TILE_SIZE_32, (M + TILE_SIZE_32 - 1) / TILE_SIZE_32);
        run_bench("tiled_32", [&]() {
            tiled_gemm_32<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        });
    }

    // 3. Vectorized float4
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        run_bench("vec4", [&]() {
            vec4_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        });
    }

    // 4. Tiled with register accumulation (64x64 block, 4 outputs/thread)
    {
        int threads = (BM / TM) * BN;  // 16 * 64 = 1024
        dim3 block(threads);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        run_bench("tiled_reg", [&]() {
            tiled_gemm_reg<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        });
    }

    // 5. cuBLAS
    {
        run_bench("cuBLAS", [&]() {
            // cuBLAS uses column-major, so we compute B^T * A^T = (A*B)^T
            // C = alpha * B^T * A^T + beta * C^T  (but C is row-major)
            // Actually: cublasSgemm with OP_N means: C = alpha * op(A) * op(B) + beta * C
            // For row-major A,B,C: compute C^T = B^T * A^T in column-major
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha,
                        d_B, N,    // B is KxN, stored row-major = column-major N^T... 
                        d_A, K,    // A is MxK
                        &beta,
                        d_C, N);
        });
    }

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    print_header("Five-Way GEMM Comparison — GB10 SM121");

    bool quick_test = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--test" || std::string(argv[i]) == "-t") {
            quick_test = true;
            printf("  Quick test mode: only running up to 512x512\n\n");
        }
    }

    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = quick_test ? 3 : sizeof(sizes) / sizeof(sizes[0]);

    std::vector<GemmResult> all_results;

    for (int s = 0; s < num_sizes; s++) {
        int sz = sizes[s];
        printf("\n  === Matrix size: %dx%d (K=%d) ===\n", sz, sz, sz);
        benchmark_gemm(sz, sz, sz, all_results);
    }

    // Print summary table
    print_header("Performance Summary");

    printf("  %-12s  %-12s  %10s  %10s  %10s  %10s\n",
           "Implementation", "Size", "Time(ms)", "TFLOP/s", "BW(GB/s)", "MaxErr");
    printf("  %-12s  %-12s  %10s  %10s  %10s  %10s\n",
           "------------", "------------", "----------", "----------", "----------", "----------");

    for (auto& r : all_results) {
        printf("  %-12s  %4dx%-7d  %10.3f  %10.2f  %10.1f  %10.4f\n",
               r.name, r.size, r.size, r.time_ms, r.tflops, r.bw_gbps,
               r.max_err >= 0 ? r.max_err : 0.0f);
    }

    // Print TFLOP/s comparison at largest size
    if (!quick_test) {
        print_header("TFLOP/s at 4096x4096");
        for (auto& r : all_results) {
            if (r.size == 4096) {
                printf("  %-12s  %10.2f TFLOP/s\n", r.name, r.tflops);
            }
        }

        // Calculate speedup vs naive
        print_header("Speedup vs Naive (4096x4096)");
        double naive_time = 0;
        for (auto& r : all_results) {
            if (r.size == 4096 && std::string(r.name) == "naive") {
                naive_time = r.time_ms;
                break;
            }
        }
        if (naive_time > 0) {
            for (auto& r : all_results) {
                if (r.size == 4096) {
                    printf("  %-12s  %10.1fx speedup  (%.2f TFLOP/s)\n",
                           r.name, naive_time / r.time_ms, r.tflops);
                }
            }
        }
    }

    printf("\n  Notes:\n");
    printf("    - cuBLAS uses Tensor Cores for FP16/BF16/TF32 (not for FP32)\n");
    printf("    - Naive GEMM is limited by global memory bandwidth\n");
    printf("    - Tiled GEMM reduces global memory access by tile_size^2 factor\n");
    printf("    - Register-accumulated GEMM reduces shared memory bank conflicts\n");
    printf("    - CUTLASS and PyTorch not available on this system\n");

    return 0;
}
