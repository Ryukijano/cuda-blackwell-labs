// Project 06: Precision Lab (FP32 → FP4)
// Phase 2 — Compiler and SM Literacy
//
// Benchmark GEMM and key neural network operations across all supported
// precisions on GB10 SM121: FP32, TF32, FP16, BF16, FP8 (if supported).
//
// Build:  make
// Run:    make run

#include "cuda_utils.h"
#include "benchmark.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// Precision type traits
// ============================================================================

// FP8 is available in CUDA 13 on Blackwell
#if __CUDA_ARCH__ >= 1000
#include <cuda_fp8.h>
#define HAS_FP8 1
#else
#define HAS_FP8 0
#endif

// ============================================================================
// GEMM kernels for different precisions
// ============================================================================

// FP32 naive GEMM
__global__ void gemm_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// FP16 naive GEMM
__global__ void gemm_fp16(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        __half sum = __float2half(0.0f);
        for (int k = 0; k < K; k++) {
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        }
        C[row * N + col] = sum;
    }
}

// BF16 naive GEMM
__global__ void gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B,
                          __nv_bfloat16* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

// ============================================================================
// cuBLAS GEMM wrappers for different precisions
// ============================================================================

double cublas_gemm_fp32(cublasHandle_t handle, const float* A, const float* B, float* C,
                        int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < 3; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    cudaDeviceSynchronize();

    GpuTimer timer;
    std::vector<double> times(10);
    for (int i = 0; i < 10; i++) {
        timer.start();
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
        timer.stop();
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    return times[5];
}

double cublas_gemm_fp16(cublasHandle_t handle, const __half* A, const __half* B, __half* C,
                        int M, int N, int K) {
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    for (int i = 0; i < 3; i++)
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    cudaDeviceSynchronize();

    GpuTimer timer;
    std::vector<double> times(10);
    for (int i = 0; i < 10; i++) {
        timer.start();
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
        timer.stop();
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    return times[5];
}

double cublas_gemm_bf16(cublasHandle_t handle, const __nv_bfloat16* A, const __nv_bfloat16* B,
                        __nv_bfloat16* C, int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    // Use cublasGemmEx with BF16
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B, CUDA_R_16BF, N,
                 A, CUDA_R_16BF, K,
                 &beta,
                 C, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    for (int i = 0; i < 3; i++)
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta,
                     C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    GpuTimer timer;
    std::vector<double> times(10);
    for (int i = 0; i < 10; i++) {
        timer.start();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta,
                     C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        timer.stop();
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    return times[5];
}

// ============================================================================
// cuBLASLt narrow-precision GEMM wrappers
// ============================================================================

__global__ void fill_fp8(__nv_fp8_e4m3* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __nv_fp8_e4m3(1.0f);
}

__global__ void fill_fp4x2(__nv_fp4x2_e2m1* out, int pairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) out[i] = __nv_fp4x2_e2m1(make_float2(1.0f, 1.0f));
}

static size_t round_off(size_t x, size_t granul) {
    return granul * ((x + (granul - 1)) / granul);
}

static size_t scale_tensor_size_vec16(int inner, int outer) {
    const size_t S_VSCALE = 16;
    const size_t S_BLOCK_COLS = 32;
    const size_t S_BLOCK_ROWS = 4;
    const size_t S_BLOCK_INNER = 4;
    const size_t BLOCK_ROWS = S_BLOCK_INNER * S_VSCALE;
    const size_t BLOCK_COLS = S_BLOCK_COLS * S_BLOCK_ROWS;
    size_t s_rows = round_off((size_t)inner, BLOCK_ROWS) / S_VSCALE;
    size_t s_cols = round_off((size_t)outer, BLOCK_COLS);
    return s_rows * s_cols;
}

double cublaslt_gemm_fp8(cublasLtHandle_t lt, int M, int N, int K) {
    __nv_fp8_e4m3 *d_A, *d_B;
    float *d_C, *d_scale;
    void *d_ws;
    size_t wsSize = 4 * 1024 * 1024;

    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ws, wsSize));

    float one = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_scale, &one, sizeof(float), cudaMemcpyHostToDevice));
    fill_fp8<<<(M * K + 255) / 256, 256>>>(d_A, M * K);
    fill_fp8<<<(K * N + 255) / 256, 256>>>(d_B, K * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale, sizeof(d_scale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale, sizeof(d_scale)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, M, K, M));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Adesc, Bdesc, Cdesc, Cdesc, pref, 1, &heuristic, &returned));

    if (returned == 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatmulDescDestroy(opDesc);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_scale); cudaFree(d_ws);
        return -1.0;
    }

    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < 3; i++) {
        CUBLAS_CHECK(cublasLtMatmul(lt, opDesc, &alpha, d_A, Adesc, d_B, Bdesc,
                                    &beta, d_C, Cdesc, d_C, Cdesc,
                                    &heuristic.algo, d_ws, wsSize, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    std::vector<double> times(10);
    for (int i = 0; i < 10; i++) {
        timer.start();
        CUBLAS_CHECK(cublasLtMatmul(lt, opDesc, &alpha, d_A, Adesc, d_B, Bdesc,
                                    &beta, d_C, Cdesc, d_C, Cdesc,
                                    &heuristic.algo, d_ws, wsSize, 0));
        timer.stop();
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    double ms = times[5];

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(opDesc);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_scale); cudaFree(d_ws);
    return ms;
}

double cublaslt_gemm_fp4(cublasLtHandle_t lt, int M, int N, int K) {
    int pairsA = (M * K + 1) / 2;
    int pairsB = (K * N + 1) / 2;
    int pairsD = (M * N + 1) / 2;

    __nv_fp4x2_e2m1 *d_A = nullptr, *d_B = nullptr, *d_D = nullptr;
    __nv_bfloat16 *d_C = nullptr;
    __nv_fp8_e4m3 *d_scaleA = nullptr, *d_scaleB = nullptr, *d_out_scale = nullptr;
    float *d_scaleD = nullptr;
    void *d_ws = nullptr;
    size_t wsSize = 4 * 1024 * 1024;

    CUDA_CHECK(cudaMalloc(&d_A, (size_t)pairsA * sizeof(__nv_fp4x2_e2m1)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)pairsB * sizeof(__nv_fp4x2_e2m1)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_D, (size_t)pairsD * sizeof(__nv_fp4x2_e2m1)));
    CUDA_CHECK(cudaMalloc(&d_ws, wsSize));

    cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

    size_t scaleA_elems = scale_tensor_size_vec16(K, M);
    size_t scaleB_elems = scale_tensor_size_vec16(K, N);
    size_t scaleDOut_elems = scale_tensor_size_vec16(M, N);
    CUDA_CHECK(cudaMalloc(&d_scaleA, scaleA_elems * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_scaleB, scaleB_elems * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_out_scale, scaleDOut_elems * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_scaleD, sizeof(float)));

    fill_fp4x2<<<(pairsA + 255) / 256, 256>>>(d_A, pairsA);
    fill_fp4x2<<<(pairsB + 255) / 256, 256>>>(d_B, pairsB);
    CUDA_CHECK(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__nv_bfloat16)));
    fill_fp8<<<(scaleA_elems + 255) / 256, 256>>>(d_scaleA, scaleA_elems);
    fill_fp8<<<(scaleB_elems + 255) / 256, 256>>>(d_scaleB, scaleB_elems);
    fill_fp8<<<(scaleDOut_elems + 255) / 256, 256>>>(d_out_scale, scaleDOut_elems);
    float one = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_scaleD, &one, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned = 0;
    cublasStatus_t st;
    double ms = -1.0;
    cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
    float alpha = 1.0f, beta = 0.0f;
    GpuTimer timer;
    std::vector<double> times(10);

    st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) goto cleanup;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode, sizeof(AScaleMode));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode, sizeof(BScaleMode));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &DScaleMode, sizeof(DScaleMode));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &DOutScaleMode, sizeof(DOutScaleMode));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scaleD, sizeof(d_scaleD));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_out_scale, sizeof(d_out_scale));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, M));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_4F_E2M1, M, N, M));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));

    st = cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &heuristic, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        goto cleanup;
    }

    for (int i = 0; i < 3; i++) {
        st = cublasLtMatmul(lt, opDesc, &alpha, d_A, Adesc, d_B, Bdesc,
                            &beta, d_C, Cdesc, d_D, Ddesc,
                            &heuristic.algo, d_ws, wsSize, 0);
        if (st != CUBLAS_STATUS_SUCCESS) goto cleanup;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < 10; i++) {
        timer.start();
        st = cublasLtMatmul(lt, opDesc, &alpha, d_A, Adesc, d_B, Bdesc,
                            &beta, d_C, Cdesc, d_D, Ddesc,
                            &heuristic.algo, d_ws, wsSize, 0);
        timer.stop();
        if (st != CUBLAS_STATUS_SUCCESS) goto cleanup;
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    ms = times[5];

cleanup:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaFree(d_scaleA); cudaFree(d_scaleB); cudaFree(d_out_scale); cudaFree(d_scaleD); cudaFree(d_ws);
    return ms;
}

// ============================================================================
// Pointwise operations (ReLU, GELU, SiLU) in different precisions
// ============================================================================

__global__ void relu_fp32(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(in[i], 0.0f);
}

__global__ void relu_fp16(const __half* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hmax(in[i], __float2half(0.0f));
}

__global__ void silu_fp32(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / (1.0f + expf(-in[i]));
}

__global__ void gelu_fp32(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + erff(x / 1.41421356f));
    }
}

// ============================================================================
// Reduction kernel
// ============================================================================

__global__ void reduce_sum_fp32(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// ============================================================================
// Softmax kernel
// ============================================================================

__global__ void softmax_fp32(const float* in, float* out, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    // Find max (for numerical stability)
    __shared__ float s_max;
    __shared__ float s_sum;
    if (tid == 0) s_max = -INFINITY;
    __syncthreads();

    for (int j = tid; j < N; j += blockDim.x) {
        float val = in[row * N + j];
        atomicMax((int*)&s_max, __float_as_int(fmaxf(val, s_max)));
    }
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float val = expf(in[row * N + j] - s_max);
        out[row * N + j] = val;
        local_sum += val;
    }

    // Reduce sum
    __shared__ float s_partial[1024];
    s_partial[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_partial[tid] += s_partial[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_sum = s_partial[0];
    __syncthreads();

    // Normalize
    for (int j = tid; j < N; j += blockDim.x) {
        out[row * N + j] /= s_sum;
    }
}

// ============================================================================
// Benchmark functions
// ============================================================================

void benchmark_gemm_precisions(int M, int N, int K) {
    printf("\n  --- GEMM %dx%dx%d ---\n", M, N, K);
    double flops = 2.0 * M * N * K;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasLtHandle_t lt;
    cublasLtCreate(&lt);

    // FP32
    {
        size_t bytes = (size_t)M * K * sizeof(float) + K * N * sizeof(float) + M * N * sizeof(float);
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        cudaMemset(d_A, 0, M * K * sizeof(float));
        cudaMemset(d_B, 0, K * N * sizeof(float));

        double ms = cublas_gemm_fp32(handle, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  %-10s  cuBLAS  time=%8.3f ms  TFLOP/s=%6.2f  BW=%5.1f GB/s  bytes=%zu\n",
               "FP32", ms, tflops, bytes / (ms * 1e-3) / 1e9, bytes);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    // FP16
    {
        size_t bytes = (size_t)M * K * sizeof(__half) + K * N * sizeof(__half) + M * N * sizeof(__half);
        __half *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));
        cudaMemset(d_A, 0, M * K * sizeof(__half));
        cudaMemset(d_B, 0, K * N * sizeof(__half));

        double ms = cublas_gemm_fp16(handle, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  %-10s  cuBLAS  time=%8.3f ms  TFLOP/s=%6.2f  BW=%5.1f GB/s  bytes=%zu\n",
               "FP16", ms, tflops, bytes / (ms * 1e-3) / 1e9, bytes);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    // BF16
    {
        size_t bytes = (size_t)M * K * sizeof(__nv_bfloat16) + K * N * sizeof(__nv_bfloat16) + M * N * sizeof(__nv_bfloat16);
        __nv_bfloat16 *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16)));
        cudaMemset(d_A, 0, M * K * sizeof(__nv_bfloat16));
        cudaMemset(d_B, 0, K * N * sizeof(__nv_bfloat16));

        double ms = cublas_gemm_bf16(handle, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  %-10s  cuBLAS  time=%8.3f ms  TFLOP/s=%6.2f  BW=%5.1f GB/s  bytes=%zu\n",
               "BF16", ms, tflops, bytes / (ms * 1e-3) / 1e9, bytes);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    // Check TF32 support (SM121 Blackwell supports TF32)
    int supports_tf32 = 1;  // SM121 has TF32 Tensor Cores
    printf("  TF32 supported: %s\n", supports_tf32 ? "Yes" : "No");

    if (supports_tf32) {
        // TF32 is controlled by cublas math mode
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        size_t bytes = (size_t)M * K * sizeof(float) + K * N * sizeof(float) + M * N * sizeof(float);
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        cudaMemset(d_A, 0, M * K * sizeof(float));
        cudaMemset(d_B, 0, K * N * sizeof(float));

        double ms = cublas_gemm_fp32(handle, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  %-10s  cuBLAS  time=%8.3f ms  TFLOP/s=%6.2f  BW=%5.1f GB/s  (TF32 mode)\n",
               "TF32", ms, tflops, bytes / (ms * 1e-3) / 1e9);

        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    // FP8 E4M3 (cuBLASLt)
    {
        size_t bytes = (size_t)M * K * 1 + K * N * 1 + M * N * sizeof(float);
        double ms = cublaslt_gemm_fp8(lt, M, N, K);
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  %-10s  cuBLASLt time=%8.3f ms  TFLOP/s=%6.2f  BW=%5.1f GB/s  bytes=%zu\n",
               "FP8_E4M3", ms, tflops, bytes / (ms * 1e-3) / 1e9, bytes);
    }

    // NVFP4 E2M1 (cuBLASLt)
    {
        size_t bytes = (size_t)(M * K + 1) / 2 + (K * N + 1) / 2 + (M * N + 1) / 2;
        double ms = cublaslt_gemm_fp4(lt, M, N, K);
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  %-10s  cuBLASLt time=%8.3f ms  TFLOP/s=%6.2f  BW=%5.1f GB/s  bytes=%zu\n",
               "NVFP4_E2M1", ms, tflops, bytes / (ms * 1e-3) / 1e9, bytes);
    }

    cublasDestroy(handle);
    cublasLtDestroy(lt);
}

void benchmark_pointwise(int n) {
    printf("\n  --- Pointwise Operations (n=%d) ---\n", n);
    size_t bytes = n * sizeof(float);
    int block = 256;
    int grid = (n + block - 1) / block;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaMemset(d_in, 0, bytes);

    auto bench = [&](auto kernel, const char* name, size_t bytes_moved) {
        for (int i = 0; i < 3; i++) kernel<<<grid, block>>>(d_in, d_out, n);
        cudaDeviceSynchronize();
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            timer.start();
            kernel<<<grid, block>>>(d_in, d_out, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double ms = times[10];
        double bw = bytes_moved / (ms * 1e-3) / 1e9;
        printf("  %-15s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
               name, ms, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);
    };

    bench(relu_fp32, "ReLU (FP32)", bytes * 2);
    bench(silu_fp32, "SiLU (FP32)", bytes * 2);
    bench(gelu_fp32, "GELU (FP32)", bytes * 2);
}

void benchmark_reduction(int n) {
    printf("\n  --- Reduction (n=%d) ---\n", n);
    size_t bytes = n * sizeof(float);
    int block = 256;
    int grid = (n + block - 1) / block;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    cudaMemset(d_in, 0, bytes);

    float zero = 0.0f;
    for (int i = 0; i < 3; i++) {
        cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice);
        reduce_sum_fp32<<<grid, block, block * sizeof(float)>>>(d_in, d_out, n);
    }
    cudaDeviceSynchronize();

    GpuTimer timer;
    std::vector<double> times(20);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice);
        timer.start();
        reduce_sum_fp32<<<grid, block, block * sizeof(float)>>>(d_in, d_out, n);
        timer.stop();
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    double ms = times[10];
    double bw = bytes / (ms * 1e-3) / 1e9;
    printf("  %-15s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
           "sum (FP32)", ms, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

void benchmark_softmax(int M, int N) {
    printf("\n  --- Softmax (M=%d, N=%d) ---\n", M, N);
    size_t bytes = (size_t)M * N * sizeof(float);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaMemset(d_in, 0, bytes);

    int block = 256;
    for (int i = 0; i < 3; i++) softmax_fp32<<<M, block>>>(d_in, d_out, M, N);
    cudaDeviceSynchronize();

    GpuTimer timer;
    std::vector<double> times(20);
    for (int i = 0; i < 20; i++) {
        timer.start();
        softmax_fp32<<<M, block>>>(d_in, d_out, M, N);
        timer.stop();
        times[i] = timer.milliseconds();
    }
    std::sort(times.begin(), times.end());
    double ms = times[10];
    double bw = bytes * 2 / (ms * 1e-3) / 1e9;  // read + write
    printf("  %-15s  time=%8.3f ms  BW=%6.1f GB/s  (%4.1f%% peak)\n",
           "softmax (FP32)", ms, bw, 100.0 * bw / GB10_PEAK_BW_GBPS);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// Memory footprint comparison
// ============================================================================

void memory_footprint_table() {
    print_header("Memory Footprint by Precision (per element)");

    printf("  %-10s  %12s  %12s  %12s\n", "Type", "Bytes/elem", "Range", "Precision");
    printf("  %-10s  %12s  %12s  %12s\n", "----------", "----------", "----------", "----------");
    printf("  %-10s  %12d  %12s  %12s\n", "FP32", 4, "±3.4e38", "7 digits");
    printf("  %-10s  %12d  %12s  %12s\n", "TF32", 4, "±3.4e38", "3 digits (10 mantissa bits)");
    printf("  %-10s  %12d  %12s  %12s\n", "FP16", 2, "±65504", "3-4 digits");
    printf("  %-10s  %12d  %12s  %12s\n", "BF16", 2, "±3.4e38", "2-3 digits");
    printf("  %-10s  %12d  %12s  %12s\n", "FP8 E4M3", 1, "±448", "1-2 digits");
    printf("  %-10s  %12d  %12s  %12s\n", "FP8 E5M2", 1, "±57344", "0-1 digits");
    printf("  %-10s  %12d  %12s  %12s\n", "INT8", 1, "±128", "exact");
    printf("  %-10s  %12.1f  %12s  %12s\n", "FP4 E2M1", 0.5, "±6", "0-1 digits");
    printf("\n  Bandwidth savings (vs FP32):\n");
    printf("    FP16/BF16: 2x less memory → 2x faster for memory-bound ops\n");
    printf("    FP8:       4x less memory → 4x faster for memory-bound ops\n");
    printf("    FP4:       8x less memory → 8x faster (if supported)\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_header("Precision Lab — GB10 SM121");

    // Check supported precisions (SM121 Blackwell supports TF32 and FP8)
    int tf32_supported = 1, fp8_supported = 1;

    printf("  Precision support:\n");
    printf("    FP32:  Yes (always)\n");
    printf("    TF32:  %s\n", tf32_supported ? "Yes" : "No");
    printf("    FP16:  Yes\n");
    printf("    BF16:  Yes\n");
    printf("    FP8:   %s\n", fp8_supported ? "Yes" : "No");
    printf("    FP4:   Yes (cuBLASLt VEC16 block-scaled)\n\n");

    // Memory footprint
    memory_footprint_table();

    // GEMM benchmarks at different sizes
    print_header("GEMM Precision Comparison (cuBLAS)");

    benchmark_gemm_precisions(512, 512, 512);
    benchmark_gemm_precisions(1024, 1024, 1024);
    benchmark_gemm_precisions(2048, 2048, 2048);
    benchmark_gemm_precisions(4096, 4096, 4096);

    // Pointwise operations
    print_header("Pointwise Operations (FP32, n=16M)");
    benchmark_pointwise(16777216);

    // Reduction
    print_header("Reduction (FP32)");
    benchmark_reduction(16777216);

    // Softmax
    print_header("Softmax (FP32)");
    benchmark_softmax(1024, 1024);
    benchmark_softmax(4096, 4096);

    // Summary
    print_header("Summary");
    printf("  Key findings:\n");
    printf("    1. FP16/BF16 GEMM should be ~2x faster than FP32 (Tensor Cores)\n");
    printf("    2. TF32 GEMM should be ~8x faster than FP32 (Tensor Cores, same memory)\n");
    printf("    3. FP8 GEMM should be ~4x faster than FP16 (if supported)\n");
    printf("    4. Pointwise ops are memory-bound → precision affects bandwidth, not compute\n");
    printf("    5. Reduction is memory-bound → same as pointwise\n");
    printf("    6. Softmax involves expf → compute-bound for small N, memory-bound for large N\n\n");

    printf("  Precision selection guide:\n");
    printf("    FP32:  Use for accumulation, error-sensitive ops, final output\n");
    printf("    TF32:  Use for GEMM when you need FP32 range but can tolerate less precision\n");
    printf("    FP16:  Use for inference, mixed precision training (with loss scaling)\n");
    printf("    BF16:  Use for training (no loss scaling needed, same range as FP32)\n");
    printf("    FP8:   Use for inference (quantized weights/activations)\n");
    printf("    FP4:   Experimental inference only — verify correctness carefully on SM121\n");

    return 0;
}
