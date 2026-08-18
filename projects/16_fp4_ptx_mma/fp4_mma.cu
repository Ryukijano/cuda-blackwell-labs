// Project 16: FP4 PTX mma.sync probe
//
// Attempts to execute the native consumer-Blackwell FP4 warp-level MMA
// instruction via inline PTX:
//
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
//     .f32.e2m1.e2m1.f32.ue4m3
//
// This instruction requires PTX 9.1+ and a CUDA 13.1+ driver to launch.
// On CUDA 13.0 / driver 580.x it may compile but fail to load.

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>

// A warp-synchronous kernel that executes a single m16n8k64 block-scaled FP4 MMA.
// A/B and scale registers are all zero for this probe; a real kernel would pack
// the fragments according to the lane layout in the PTX ISA.
__global__ void fp4_mma_probe(float* out) {
    if (threadIdx.x >= 32) return;

    // 0x2 = e2m1 value 1.0; 0x38 = ue4m3 scale value 1.0.
    unsigned A[4] = {0x22222222, 0x22222222, 0x22222222, 0x22222222};
    unsigned B[2] = {0x22222222, 0x22222222};
    float D[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned sa = 0x38383838;
    unsigned sb = 0x38383838;

    // C is aliased to D, matching the common accumulate-in-place pattern.
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};"
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(sa), "r"(sb)
    );

    // Have lane 0 write the first four accumulators back to gmem.
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; i++) {
            out[i] = D[i];
        }
    }
}

int main() {
    print_header("FP4 PTX mma.sync probe — GB10 SM121");

    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(float)));

    // Launch exactly one warp.
    fp4_mma_probe<<<1, 32>>>(d_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  Kernel launch / PTX load failed: %s\n", cudaGetErrorString(err));
        printf("  This is expected on CUDA 13.0 / driver 580.x because the FP4\n");
        printf("  mma.sync PTX variant needs a 13.1+ driver.\n");
        cudaFree(d_out);
        return 1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float h[4];
    CUDA_CHECK(cudaMemcpy(h, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  FP4 mma.sync probe executed successfully.\n");
    printf("  Accumulators (lane 0): %f %f %f %f\n", h[0], h[1], h[2], h[3]);
    printf("  Expected: ~64.0 if A=B=1.0 and scales=1.0 for a m16n8k64.\n");

    cudaFree(d_out);
    return 0;
}
