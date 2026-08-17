// Project 03: CUDA → PTX → SASS Pipeline
// Phase 1 — Hardware and Memory Literacy
//
// Trace CUDA source code through the compilation pipeline:
//   .cu source → PTX (virtual ISA) → SASS (machine code for SM121)
//
// This file contains multiple kernel variants to compare compiler output.
// The Makefile generates PTX and SASS for each variant.

#include "cuda_utils.h"
#include <vector>
#include <algorithm>
#include <string>

// ============================================================================
// Variant 1: Basic SAXPY (no optimization hints)
// ============================================================================

__global__ void saxpy_basic(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// ============================================================================
// Variant 2: SAXPY with __restrict__ (aliasing hint)
// ============================================================================

__global__ void saxpy_restrict(const float* __restrict__ x,
                                float* __restrict__ y,
                                float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// ============================================================================
// Variant 3: SAXPY with float4 vectorized loads
// ============================================================================

__global__ void saxpy_vec4(const float* __restrict__ x,
                            float* __restrict__ y,
                            float a, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 xv = reinterpret_cast<const float4*>(x)[i / 4];
        float4 yv = reinterpret_cast<float4*>(y)[i / 4];
        yv.x = a * xv.x + yv.x;
        yv.y = a * xv.y + yv.y;
        yv.z = a * xv.z + yv.z;
        yv.w = a * xv.w + yv.w;
        reinterpret_cast<float4*>(y)[i / 4] = yv;
    }
}

// ============================================================================
// Variant 4: SAXPY with shared memory tiling
// ============================================================================

__global__ void saxpy_shared(const float* __restrict__ x,
                              float* __restrict__ y,
                              float a, int n) {
    extern __shared__ float s_x[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load tile into shared memory
    if (i < n) {
        s_x[tid] = x[i];
    }
    __syncthreads();

    if (i < n) {
        y[i] = a * s_x[tid] + y[i];
    }
}

// ============================================================================
// Variant 5: SAXPY with __launch_bounds__
// ============================================================================

__global__ void __launch_bounds__(256, 4)
saxpy_launchbounds(const float* __restrict__ x,
                    float* __restrict__ y,
                    float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// ============================================================================
// Variant 6: SAXPY with manual loop unrolling
// ============================================================================

__global__ void saxpy_unrolled(const float* __restrict__ x,
                                float* __restrict__ y,
                                float a, int n) {
    int base = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int i = base + j;
        if (i < n) {
            y[i] = a * x[i] + y[i];
        }
    }
}

// ============================================================================
// Variant 7: Reduction kernel (different pattern for SASS comparison)
// ============================================================================

__global__ void reduce_sum(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// ============================================================================
// Host code: launch all variants and verify correctness
// ============================================================================

bool verify_saxpy(const float* x, const float* y_orig, float a, int n, const float* y_result) {
    for (int i = 0; i < n; i++) {
        float expected = a * x[i] + y_orig[i];
        if (fabsf(y_result[i] - expected) > 1e-5) {
            printf("  MISMATCH at %d: got %f, expected %f\n", i, y_result[i], expected);
            return false;
        }
    }
    return true;
}

int main() {
    print_header("CUDA → PTX → SASS Pipeline — GB10 SM121");

    int n = 1 << 20;  // 1M elements
    size_t bytes = n * sizeof(float);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;

    // Allocate host arrays
    std::vector<float> h_x(n), h_y(n), h_y_orig(n);
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)i * 0.001f;
        h_y[i] = 1.0f;
        h_y_orig[i] = 1.0f;
    }

    // Allocate device arrays
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    float a = 2.0f;

    // Test each variant
    struct { const char* name; void (*kernel)(const float*, float*, float, int); size_t shared; } variants[] = {
        {"saxpy_basic",        saxpy_basic,        0},
        {"saxpy_restrict",     saxpy_restrict,     0},
        {"saxpy_vec4",         saxpy_vec4,         0},
        {"saxpy_shared",       saxpy_shared,       block_size * sizeof(float)},
        {"saxpy_launchbounds", saxpy_launchbounds, 0},
        {"saxpy_unrolled",     saxpy_unrolled,     0},
    };

    printf("  %-25s  %12s  %12s  %12s\n", "Kernel", "Time (ms)", "BW (GB/s)", "Correct?");
    printf("  %-25s  %12s  %12s  %12s\n", "-------------------------", "----------", "----------", "---------");

    for (auto& v : variants) {
        // Reset y for correctness verification (single run)
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

        int grid_v = (v.name == std::string("saxpy_vec4") || v.name == std::string("saxpy_unrolled"))
                     ? (n + block_size * 4 - 1) / (block_size * 4) : grid;

        // Single run for correctness
        if (v.shared > 0)
            v.kernel<<<grid_v, block_size, v.shared>>>(d_x, d_y, a, n);
        else
            v.kernel<<<grid_v, block_size>>>(d_x, d_y, a, n);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        bool correct = verify_saxpy(h_x.data(), h_y_orig.data(), a, n, h_y.data());

        // Now benchmark (don't care about correctness, just timing)
        // Use a dummy output to avoid accumulation
        float *d_out;
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        cudaMemset(d_out, 0, bytes);

        // Warmup
        for (int i = 0; i < 3; i++) {
            if (v.shared > 0)
                v.kernel<<<grid_v, block_size, v.shared>>>(d_x, d_out, a, n);
            else
                v.kernel<<<grid_v, block_size>>>(d_x, d_out, a, n);
        }
        cudaDeviceSynchronize();

        // Benchmark
        GpuTimer timer;
        std::vector<double> times(20);
        for (int i = 0; i < 20; i++) {
            // Reset output each time to avoid accumulation
            cudaMemset(d_out, 0, bytes);
            timer.start();
            if (v.shared > 0)
                v.kernel<<<grid_v, block_size, v.shared>>>(d_x, d_out, a, n);
            else
                v.kernel<<<grid_v, block_size>>>(d_x, d_out, a, n);
            timer.stop();
            times[i] = timer.milliseconds();
        }
        std::sort(times.begin(), times.end());
        double median = times[10];
        double bw = effective_bandwidth_gbps(bytes * 3, median);  // read x + read y + write y

        printf("  %-25s  %10.3f      %10.1f    %s\n",
               v.name, median, bw, correct ? "OK" : "FAIL");

        CUDA_CHECK(cudaFree(d_out));
        // Reset for next variant
        std::copy(h_y_orig.begin(), h_y_orig.end(), h_y.begin());
    }

    // Print compilation instructions
    print_header("PTX and SASS Inspection Commands");

    printf("  The Makefile generates PTX and SASS for each variant.\n\n");

    printf("  Generate PTX (virtual ISA):\n");
    printf("    nvcc -arch=compute_121 -ptx -O2 pipeline.cu -o pipeline.ptx\n\n");

    printf("  Generate SASS (machine code):\n");
    printf("    cuobjdump --dump-sass pipeline_O3 > sass_O3.txt\n");
    printf("    cuobjdump --dump-sass pipeline_O0 > sass_O0.txt\n\n");

    printf("  Compare O0 vs O3:\n");
    printf("    diff sass_O0.txt sass_O3.txt\n\n");

    printf("  Inspect with line info (maps SASS to source):\n");
    printf("    nvdisasm --print-line-info pipeline_O3 > sass_lineinfo.txt\n\n");

    printf("  Compare architectures:\n");
    printf("    nvcc -arch=sm_80  -lineinfo -O3 pipeline.cu -o pipeline_sm80\n");
    printf("    nvcc -arch=sm_90  -lineinfo -O3 pipeline.cu -o pipeline_sm90\n");
    printf("    nvcc -arch=sm_121 -lineinfo -O3 pipeline.cu -o pipeline_sm121\n");
    printf("    cuobjdump --dump-sass pipeline_sm80  > sass_sm80.txt\n");
    printf("    cuobjdump --dump-sass pipeline_sm90  > sass_sm90.txt\n");
    printf("    cuobjdump --dump-sass pipeline_sm121 > sass_sm121.txt\n\n");

    printf("  Key things to look for in SASS:\n");
    printf("    LDG.E.32 / LDG.E.64 / LDG.E.128  — load width (vectorization)\n");
    printf("    STG.E.32 / STG.E.128             — store width\n");
    printf("    FFMA / FMUL+FADD                 — fused multiply-add vs separate\n");
    printf("    ISETP / @P0                      — predicated execution\n");
    printf("    BRA                              — branch (not predicated)\n");
    printf("    IMAD / IADD3                     — integer multiply-add / add\n");
    printf("    EXIT                             — kernel exit\n");
    printf("    R0-R255                          — general registers\n");
    printf("    UR0-UR255                        — uniform registers\n\n");

    printf("  Run 'make sass' to generate all SASS files automatically.\n");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}
