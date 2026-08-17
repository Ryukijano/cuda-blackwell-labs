// Project 06 add-on: FP8 and NVFP4 cublasLt checks
//
// Uses cuBLASLt to exercise the Blackwell narrow-precision Tensor Core paths.
// FP8 (E4M3) should work on SM121.  NVFP4 (E2M1) is attempted; on this
// CUDA 13.0 / cuBLAS 13.1 setup it currently fails heuristic selection and
// is reported as unsupported rather than hidden.

#include "cuda_utils.h"
#include "benchmark.h"
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

__global__ void fill_fp8(__nv_fp8_e4m3* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __nv_fp8_e4m3(1.0f);
}

__global__ void fill_fp4x2(__nv_fp4x2_e2m1* out, int pairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) out[i] = __nv_fp4x2_e2m1(make_float2(1.0f, 1.0f));
}

// Round up to next multiple
static size_t round_off(size_t x, size_t granul) {
    return granul * ((x + (granul - 1)) / granul);
}

// cuBLASLt scale-tensor size for VEC16_UE4M3 (used for FP4 block scales)
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

// ----------------------------------------------------------------------------
// FP8 E4M3 GEMM via cuBLASLt
// ----------------------------------------------------------------------------

double cublaslt_gemm_fp8(cublasLtHandle_t lt, int M, int N, int K) {
    double flops = 2.0 * M * N * K;

    __nv_fp8_e4m3 *d_A, *d_B;
    float *d_C;
    float *d_scale;
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
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
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
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
        CUBLAS_CHECK(cublasLtMatmulDescDestroy(opDesc));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_scale); cudaFree(d_ws);
        return -1.0;
    }

    // Warmup
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

    // Sanity check: C[0] should be K (all 1.0 inputs)
    float c0;
    CUDA_CHECK(cudaMemcpy(&c0, d_C, sizeof(float), cudaMemcpyDeviceToHost));
    if (fabsf(c0 - (float)K) > 1.0f) {
        printf("    FP8 sanity mismatch: C[0]=%.3f, expected %d\n", c0, K);
    }

    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(opDesc));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_scale); cudaFree(d_ws);

    return ms;
}

// ----------------------------------------------------------------------------
// NVFP4 E2M1 GEMM attempt via cuBLASLt
// ----------------------------------------------------------------------------

double cublaslt_gemm_fp4_attempt(cublasLtHandle_t lt, int M, int N, int K) {
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

    // A/B/DOut use VEC16_UE4M3 block scales; D (the FP4 input/output scale)
    // uses a per-tensor float scalar.  All-1.0 keeps magnitudes unchanged.
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

    // With opA=T, A is K x M in col-major; with opB=N, B is K x N in col-major.
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

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    print_header("Narrow Precision Check — FP8 / NVFP4 — GB10 SM121");

    bool quick = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--test" || std::string(argv[i]) == "-t") {
            quick = true;
            printf("  Quick test mode: only 512x512\n\n");
        }
    }

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    int sizes[] = {512, 1024, 2048, 4096};
    int num = quick ? 1 : (sizeof(sizes) / sizeof(sizes[0]));

    printf("  %-10s  %-10s  %10s  %10s  %10s  %10s\n",
           "Size", "Type", "Time(ms)", "TFLOP/s", "BW(GB/s)", "Status");
    printf("  %-10s  %-10s  %10s  %10s  %10s  %10s\n",
           "----------", "----------", "----------", "----------", "----------", "----------");

    for (int s = 0; s < num; s++) {
        int sz = sizes[s];
        double flops = 2.0 * sz * sz * sz;

        // FP8 E4M3
        double ms_fp8 = cublaslt_gemm_fp8(lt, sz, sz, sz);
        if (ms_fp8 > 0) {
            double tflops = flops / (ms_fp8 * 1e-3) / 1e12;
            size_t bytes = (size_t)sz * sz * 1 * 2 + sz * sz * 4; // A+B FP8 + C FP32
            double bw = bytes / (ms_fp8 * 1e-3) / 1e9;
            printf("  %4dx%-5d  %-10s  %10.3f  %10.2f  %10.1f  %10s\n",
                   sz, sz, "FP8_E4M3", ms_fp8, tflops, bw, "OK");
        } else {
            printf("  %4dx%-5d  %-10s  %10s  %10s  %10s  %10s\n",
                   sz, sz, "FP8_E4M3", "-", "-", "-", "FAILED");
        }

        // FP4 attempt
        double ms_fp4 = cublaslt_gemm_fp4_attempt(lt, sz, sz, sz);
        if (ms_fp4 > 0) {
            double tflops = flops / (ms_fp4 * 1e-3) / 1e12;
            size_t bytes = (size_t)sz * sz / 2 * 2 + sz * sz * 2; // A+B FP4 + C BF16
            double bw = bytes / (ms_fp4 * 1e-3) / 1e9;
            printf("  %4dx%-5d  %-10s  %10.3f  %10.2f  %10.1f  %10s\n",
                   sz, sz, "NVFP4_E2M1", ms_fp4, tflops, bw, "OK");
        } else {
            printf("  %4dx%-5d  %-10s  %10s  %10s  %10s  %10s\n",
                   sz, sz, "NVFP4_E2M1", "-", "-", "-", "UNSUPPORTED");
        }
    }

    CUBLAS_CHECK(cublasLtDestroy(lt));
    return 0;
}
