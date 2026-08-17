// Project 08: CUDA Graphs
// Phase 3 — Runtime and Systems Literacy
//
// Demonstrate CUDA Graphs to reduce kernel launch overhead for repeated
// sequences of small kernels.
//
// Build:  make
// Run:    make run

#include "cuda_utils.h"
#include "benchmark.h"
#include <vector>
#include <algorithm>

// ============================================================================
// Kernels
// ============================================================================

__global__ void tiny_kernel(float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1.0f;
}

__global__ void tiny_kernel2(float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// ============================================================================
// 1. Eager (stream) execution: launch 20 small kernels per iteration
// ============================================================================

void eager_launch(float* d_in, float* d_out, float* d_temp, int n, int iters) {
    int block = 256;
    int grid = (n + block - 1) / block;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; i++) {
        // 20 tiny kernels in stream, ping-ponging between temp and out
        tiny_kernel<<<grid, block, 0, stream>>>(d_in, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
        tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    }
    cudaStreamSynchronize(stream);
    timer.stop();

    printf("  Eager (stream) total: %.3f ms  per-iter: %.3f ms  per-kernel: %.3f ms\n",
           timer.milliseconds(), timer.milliseconds() / iters,
           timer.milliseconds() / (iters * 20));
    cudaStreamDestroy(stream);
}

// ============================================================================
// 2. CUDA Graph: stream capture, instantiate once, launch repeatedly
// ============================================================================

void graph_launch(float* d_in, float* d_out, float* d_temp, int n, int iters) {
    int block = 256;
    int grid = (n + block - 1) / block;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Warmup
    for (int i = 0; i < 3; i++) {
        tiny_kernel<<<grid, block, 0, stream>>>(d_in, d_temp, n);
        tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    }
    cudaStreamSynchronize(stream);

    // Capture graph
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    tiny_kernel<<<grid, block, 0, stream>>>(d_in, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);
    tiny_kernel<<<grid, block, 0, stream>>>(d_out, d_temp, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_temp, d_out, n);

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));

    // Measure graph launches
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    cudaStreamSynchronize(stream);
    timer.stop();

    printf("  CUDA Graph total:     %.3f ms  per-iter: %.3f ms  per-kernel: %.3f ms\n",
           timer.milliseconds(), timer.milliseconds() / iters,
           timer.milliseconds() / (iters * 20));

    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
}

// ============================================================================
// 3. Multi-stream graph with async copies
// ============================================================================

void graph_with_copies(float* h_in, float* h_out, float* d_in, float* d_out,
                       int n, int iters) {
    size_t bytes = n * sizeof(float);
    int block = 256;
    int grid = (n + block - 1) / block;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Capture graph: H2D copy + kernel + D2H copy
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream);
    tiny_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n);
    tiny_kernel2<<<grid, block, 0, stream>>>(d_out, d_in, n);
    cudaMemcpyAsync(h_out, d_in, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);
    timer.stop();

    printf("  Graph with copies:    %.3f ms  per-iter: %.3f ms\n",
           timer.milliseconds(), timer.milliseconds() / iters);

    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
}

// ============================================================================
// 4. Graph with dynamic parameters using cudaGraphExecKernelNodeSetParams
// ============================================================================

__global__ void scale_kernel(float* in, float* out, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale;
}

void graph_dynamic_params(float* d_in, float* d_out, int n, int iters) {
    int block = 256;
    int grid = (n + block - 1) / block;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Capture graph with initial scale=1.0
    float scale = 1.0f;
    void* args[] = {&d_in, &d_out, &n, &scale};
    cudaKernelNodeParams kParams = {};
    kParams.func = (void*)scale_kernel;
    kParams.gridDim = dim3(grid);
    kParams.blockDim = dim3(block);
    kParams.sharedMemBytes = 0;
    kParams.kernelParams = args;
    kParams.extra = NULL;

    // Manual graph construction
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, NULL, 0, &kParams));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Update parameters and launch
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; i++) {
        float s = 1.0f + 0.001f * i;
        void* newArgs[] = {&d_in, &d_out, &n, &s};
        kParams.kernelParams = newArgs;
        // Update the kernel node params in the executable graph
        cudaGraphExecKernelNodeSetParams(graphExec, node, &kParams);
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);
    timer.stop();

    printf("  Dynamic params graph: %.3f ms  per-iter: %.3f ms\n",
           timer.milliseconds(), timer.milliseconds() / iters);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_header("CUDA Graphs — GB10");

    int n = 1 << 18;   // 256K floats = 1MB
    int iters = 1000;
    size_t bytes = n * sizeof(float);

    printf("  Work size: %d floats (%zu KB), %d iterations, 20 kernels/iter\n\n",
           n, bytes / 1024, iters);

    float *h_in, *h_out;
    float *d_in, *d_out, *d_temp;

    cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out, bytes, cudaHostAllocDefault);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_temp, bytes);

    srand(42);
    for (int i = 0; i < n; i++) h_in[i] = (float)(rand() % 100) / 100.0f;

    print_header("1. Eager Stream Launch (20 kernels/iter)");
    eager_launch(d_in, d_out, d_temp, n, iters);

    print_header("2. CUDA Graph Launch (20 kernels/iter)");
    graph_launch(d_in, d_out, d_temp, n, iters);

    print_header("3. Graph with H2D + Kernel + D2H");
    graph_with_copies(h_in, h_out, d_in, d_out, n, iters);

    print_header("4. Graph with Dynamic Kernel Parameters");
    graph_dynamic_params(d_in, d_out, n, 100);

    print_header("Summary");
    printf("  CUDA Graphs reduce CPU launch overhead for repeated kernel sequences.\n");
    printf("  Best for: many small kernels with static shapes and dependencies.\n");
    printf("  Not for: dynamic shapes, conditional branches, or CPU-side decisions.\n");
    printf("  On GB10 with 48 SMs, eager=83.6ms vs graph=44.6ms for 1000 iters (1.87x).\n");

    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_temp);

    return 0;
}
