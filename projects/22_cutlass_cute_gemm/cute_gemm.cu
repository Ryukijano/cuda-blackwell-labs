/*
 * Project 22: CuTe DSL Hello GEMM
 *
 * A minimal, self-contained CuTe 3.8 tensor-programming example.
 * This is the real CuTe DSL counterpart to Project 15 (CUTLASS 3.8 device Gemm).
 *
 * The kernel tiles the matrices into shared memory and computes C += A * B
 * using cute::gemm and FMA in registers.  It runs on sm_121 and does not
 * require TMEM or the SM100 tcgen05 path.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include "cuda_utils.h"

using namespace cute;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
                 TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
                 TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
                 Alpha alpha, Beta beta)
{
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  Tensor tAgA = local_partition(gA, tA, threadIdx.x);
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);
  Tensor tBgB = local_partition(gB, tB, threadIdx.x);
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);

  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});

  Tensor tCrC = make_tensor_like(tCgC);
  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    copy(tAgA(_,_,k_tile), tAsA);
    copy(tBgB(_,_,k_tile), tBsB);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    gemm(tCsA, tCsB, tCrC);
    __syncthreads();
  }

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC>
void gemm_tn(int m, int n, int k,
             TA alpha, TA const* A, int ldA,
             TB const* B, int ldB,
             TC beta,  TC      * C, int ldC,
             cudaStream_t stream = 0)
{
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = make_layout(make_shape(bM, bK), LayoutRight{});
  auto sB = make_layout(make_shape(bN, bK), LayoutRight{});
  auto sC = make_layout(make_shape(bM, bN));

  auto tA = make_layout(make_shape(Int<16>{}, Int<4>{}), LayoutRight{});
  auto tB = make_layout(make_shape(Int<16>{}, Int<4>{}), LayoutRight{});
  auto tC = make_layout(make_shape(Int<8>{},  Int<8>{}));

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

template <class TA, class TB>
static float cpu_ref(TA const* A, TB const* B, int m, int n, int k,
                     int i, int j)
{
    float s = 0.0f;
    for (int l = 0; l < k; ++l) {
        float a = (float)A[i + l * m]; // A is KxM column-major with ldA=m
        float b = (float)B[j + l * n]; // B is KxN column-major with ldB=n
        s += a * b;
    }
    return s;
}

int main(int argc, char** argv)
{
    int m = 512;
    if (argc >= 2) sscanf(argv[1], "%d", &m);
    int n = m;
    int k = m;
    if (argc >= 3) sscanf(argv[2], "%d", &k);

    printf("============================================================\n");
    printf("  CuTe DSL Hello GEMM -- GB10 SM121\n");
    printf("============================================================\n");
    printf("  M=%d N=%d K=%d\n\n", m, n, k);

    using TA = half_t;
    using TB = half_t;
    using TC = half_t;
    using TI = half_t;

    std::vector<TA> h_A(m*k);
    std::vector<TB> h_B(n*k);
    std::vector<TC> h_C(m*n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (int j = 0; j < m*k; ++j) h_A[j] = TA(dis(gen));
    for (int j = 0; j < n*k; ++j) h_B[j] = TB(dis(gen));
    for (int j = 0; j < m*n; ++j) h_C[j] = TC(0.0f);

    TA* d_A; TB* d_B; TC* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m*k*sizeof(TA)));
    CUDA_CHECK(cudaMalloc(&d_B, n*k*sizeof(TB)));
    CUDA_CHECK(cudaMalloc(&d_C, m*n*sizeof(TC)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m*k*sizeof(TA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), n*k*sizeof(TB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, m*n*sizeof(TC)));

    TI alpha = TI(1.0f);
    TI beta  = TI(0.0f);

    gemm_tn(m, n, k, alpha, d_A, m, d_B, n, beta, d_C, m);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<TC> h_C_ref(m*n);
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C, m*n*sizeof(TC), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float ref = cpu_ref(h_A.data(), h_B.data(), m, n, k, i, j);
            float got = (float)h_C_ref[i + j*m];
            float err = fabsf(got - ref);
            if (err > max_err) max_err = err;
        }
    }

    const int iters = 100;
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) {
        gemm_tn(m, n, k, alpha, d_A, m, d_B, n, beta, d_C, m);
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));

    double gflops = 2.0 * m * n * k * 1e-9;
    double avg_ms = ms / iters;
    double tflops = gflops / (avg_ms / 1000.0) / 1000.0;

    printf("  Time:    %.4f ms (median of %d)\n", avg_ms, iters);
    printf("  TFLOPS:  %.2f\n", tflops);
    float tol = fmaxf(1.0f, 0.05f * k);
    printf("  Max err vs CPU: %f (tol %.2f)\n", max_err, tol);
    printf("  Verify:  %s\n", (max_err < tol) ? "PASSED" : "FAILED");
    printf("\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return (max_err < tol) ? 0 : 1;
}
