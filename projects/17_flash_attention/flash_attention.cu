// Project 17: FlashAttention-style online softmax attention
//
// Demonstrates the online softmax trick used in FlashAttention: for each query
// row we stream over K/V, maintaining a running max and running sum so the full
// N x N attention matrix never has to be materialised.
//
// This is a correctness-first implementation (one thread per query row, serial
// over keys).  A tiled, fully parallel version would follow the same online
// update pattern with K/V tiles in shared memory and warp-level reductions.

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

#define N 256
#define D 64

__device__ float dot(const float* __restrict__ a, const float* __restrict__ b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}

__global__ void online_softmax_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float scale
) {
    int row = blockIdx.x;
    if (threadIdx.x != 0) return;  // serial-per-query correctness probe

    const float* q = Q + row * D;
    float o[D];
    for (int i = 0; i < D; i++) o[i] = 0.0f;

    float m = -1e20f;  // running max
    float l = 0.0f;    // running sum of exp(s - m)

    for (int j = 0; j < N; j++) {
        float s = dot(q, K + j * D, D) * scale;

        float m_new = fmaxf(m, s);
        float exp_m = expf(m - m_new);   // rescale previous stats
        float exp_s = expf(s - m_new);

        l = l * exp_m + exp_s;
        for (int i = 0; i < D; i++)
            o[i] = o[i] * exp_m + exp_s * V[j * D + i];

        m = m_new;
    }

    float inv_l = 1.0f / l;
    for (int i = 0; i < D; i++)
        O[row * D + i] = o[i] * inv_l;
}

void cpu_attention(const float* Q, const float* K, const float* V, float* O) {
    float scale = 1.0f / sqrtf((float)D);
    for (int row = 0; row < N; row++) {
        float max_s = -1e20f;
        std::vector<float> s(N);
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int i = 0; i < D; i++)
                sum += Q[row * D + i] * K[j * D + i];
            s[j] = sum * scale;
            max_s = fmaxf(max_s, s[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += expf(s[j] - max_s);
        float inv_l = 1.0f / sum;
        for (int i = 0; i < D; i++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += expf(s[j] - max_s) * V[j * D + i];
            O[row * D + i] = acc * inv_l;
        }
    }
}

int main() {
    print_header("FlashAttention-style online softmax attention");

    size_t n_bytes = N * D * sizeof(float);
    std::vector<float> h_Q(N * D), h_K(N * D), h_V(N * D), h_O(N * D), h_ref(N * D);

    for (size_t i = 0; i < h_Q.size(); i++) {
        h_Q[i] = (float)(rand() % 7 - 3) * 0.1f;
        h_K[i] = (float)(rand() % 7 - 3) * 0.1f;
        h_V[i] = (float)(rand() % 7 - 3) * 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, n_bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), n_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), n_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), n_bytes, cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)D);

    // Warmup
    online_softmax_attention<<<N, 1>>>(d_Q, d_K, d_V, d_O, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    GpuTimer t;
    t.start();
    online_softmax_attention<<<N, 1>>>(d_Q, d_K, d_V, d_O, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.stop();

    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, n_bytes, cudaMemcpyDeviceToHost));

    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data());

    float max_err = 0.0f;
    for (size_t i = 0; i < h_O.size(); i++)
        max_err = fmaxf(max_err, fabsf(h_O[i] - h_ref[i]));

    double flops = 2.0 * (double)N * N * D * 4;  // QK + PV + softmax (approx)
    double ms = t.milliseconds();
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("\n  N=%d, D=%d\n", N, D);
    printf("  Time:    %.4f ms\n", ms);
    printf("  TFLOPS:  %.2f\n", tflops);
    printf("  Max err vs CPU: %.6f\n", max_err);
    printf("  Verify:  %s\n", max_err < 1e-3 ? "PASSED" : "FAILED");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return max_err < 1e-3 ? 0 : 1;
}
