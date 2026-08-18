// Project 15: CUTLASS 3.8 Hello GEMM
//
// Minimal FP16 GEMM using the CUTLASS 3.8 device-level GEMM API.

#include "cuda_utils.h"

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

// Use the default CUTLASS device Gemm (SIMT by default; works on sm_121).
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor
>;

int main() {
    print_header("CUTLASS 3.8 Hello FP16 GEMM — GB10");

    int M = 1024, N = 1024, K = 1024;
    cutlass::half_t alpha = 1.0_hf;
    cutlass::half_t beta  = 0.0_hf;

    // Column-major tensors.
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C(cutlass::MatrixCoord(M, N));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> D_ref(cutlass::MatrixCoord(M, N));

    uint64_t seed = 2080;
    cutlass::reference::device::TensorFillRandomGaussian(A.device_view(), seed,        0.0_hf, 2.0_hf, 0);
    cutlass::reference::device::TensorFillRandomGaussian(B.device_view(), seed * 2019, 0.0_hf, 2.0_hf, 0);
    cutlass::reference::device::TensorFillRandomGaussian(C.device_view(), seed * 1993, 0.0_hf, 0.0_hf, 0);

    // Beta is 0, so D_ref can be zero-initialized.
    cutlass::reference::device::TensorFillRandomGaussian(D_ref.device_view(), seed * 1994, 0.0_hf, 0.0_hf, 0);

    // Run once for verification.
    Gemm gemm_op;
    cutlass::Status status = gemm_op({
        {M, N, K},
        {A.device_data(), A.stride(0)},
        {B.device_data(), B.stride(0)},
        {C.device_data(), C.stride(0)},
        {C.device_data(), C.stride(0)},
        {alpha, beta}
    });
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed" << std::endl;
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute reference.
    A.sync_host();
    B.sync_host();
    C.sync_host();

    cutlass::reference::host::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::half_t
    > gemm_ref;

    gemm_ref(
        {M, N, K},
        alpha,
        A.host_ref(),
        B.host_ref(),
        beta,
        D_ref.host_ref()
    );

    bool passed = cutlass::reference::host::TensorEquals(D_ref.host_view(), C.host_view());

    // Benchmark.
    GpuTimer t;
    std::vector<double> times;
    for (int i = 0; i < 11; i++) {
        t.start();
        gemm_op({
            {M, N, K},
            {A.device_data(), A.stride(0)},
            {B.device_data(), B.stride(0)},
            {C.device_data(), C.stride(0)},
            {C.device_data(), C.stride(0)},
            {alpha, beta}
        });
        CUDA_CHECK(cudaDeviceSynchronize());
        t.stop();
        times.push_back(t.milliseconds());
    }
    std::sort(times.begin(), times.end());
    double ms = times[times.size() / 2];

    double flops = 2.0 * (double)M * N * K;
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("\n  M=%d N=%d K=%d\n", M, N, K);
    printf("  Time:    %.4f ms (median of 11)\n", ms);
    printf("  TFLOPS:  %.2f\n", tflops);
    printf("  Verify:  %s\n", passed ? "PASSED" : "FAILED");

    return passed ? 0 : 1;
}
