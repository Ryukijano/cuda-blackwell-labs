// Project 12: CUDA Graph Conditional Nodes
//
// Demonstrates IF/ELSE and WHILE control flow inside a CUDA graph, with the
// condition set by device code using cudaGraphSetConditional.

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__global__ void set_handle_from_flag(cudaGraphConditionalHandle handle, int* flag) {
    if (threadIdx.x == 0) {
        cudaGraphSetConditional(handle, (*flag) ? 1u : 0u);
    }
}

__global__ void body_add(int* out, int add) {
    if (threadIdx.x == 0) {
        *out = add;
    }
}

__global__ void while_body(int* counter, int N, cudaGraphConditionalHandle handle) {
    if (threadIdx.x == 0) {
        int c = atomicAdd(counter, 1);
        // Continue if we haven't reached N yet; stop on next condition check.
        cudaGraphSetConditional(handle, (c + 1 < N) ? 1u : 0u);
    }
}

// ---------------------------------------------------------------------------
// IF/ELSE graph: run one of two body kernels based on a device flag.
// ---------------------------------------------------------------------------
static void test_if_else() {
    print_header("CUDA Graph Conditional: IF/ELSE");

    int *d_flag, *d_out;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    cudaGraphConditionalHandle handle;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));

    // Node A: set the conditional handle from d_flag.
    void* a_args[2] = {&handle, &d_flag};
    cudaGraphNodeParams a_params = {};
    a_params.type = cudaGraphNodeTypeKernel;
    a_params.kernel.func = (void*)set_handle_from_flag;
    a_params.kernel.gridDim = dim3(1);
    a_params.kernel.blockDim = dim3(32);
    a_params.kernel.kernelParams = a_args;

    cudaGraphNode_t nodeA;
    CUDA_CHECK(cudaGraphAddNode(&nodeA, graph, nullptr, nullptr, 0, &a_params));

    // Conditional IF/ELSE node, depends on nodeA.
    cudaGraphNodeParams c_params = {};
    c_params.type = cudaGraphNodeTypeConditional;
    c_params.conditional.handle = handle;
    c_params.conditional.type = cudaGraphCondTypeIf;
    c_params.conditional.size = 2;  // true + false bodies

    cudaGraphNode_t condNode;
    CUDA_CHECK(cudaGraphAddNode(&condNode, graph, &nodeA, nullptr, 1, &c_params));

    // The true body sets *out = 10, the false body sets *out = 20.
    int true_add = 10, false_add = 20;
    void* true_args[2] = {&d_out, &true_add};
    void* false_args[2] = {&d_out, &false_add};

    cudaGraph_t trueGraph = c_params.conditional.phGraph_out[0];
    cudaGraph_t falseGraph = c_params.conditional.phGraph_out[1];

    cudaGraphNodeParams true_params = {};
    true_params.type = cudaGraphNodeTypeKernel;
    true_params.kernel.func = (void*)body_add;
    true_params.kernel.gridDim = dim3(1);
    true_params.kernel.blockDim = dim3(32);
    true_params.kernel.kernelParams = true_args;
    cudaGraphNode_t trueNode;
    CUDA_CHECK(cudaGraphAddNode(&trueNode, trueGraph, nullptr, nullptr, 0, &true_params));

    cudaGraphNodeParams false_params = {};
    false_params.type = cudaGraphNodeTypeKernel;
    false_params.kernel.func = (void*)body_add;
    false_params.kernel.gridDim = dim3(1);
    false_params.kernel.blockDim = dim3(32);
    false_params.kernel.kernelParams = false_args;
    cudaGraphNode_t falseNode;
    CUDA_CHECK(cudaGraphAddNode(&falseNode, falseGraph, nullptr, nullptr, 0, &false_params));

    cudaGraphExec_t exec;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    int out;
    for (int flag : {1, 0}) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGraphLaunch(exec, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
        int expected = flag ? 10 : 20;
        printf("  flag=%d -> out=%d (expected %d) %s\n", flag, out, expected,
               (out == expected) ? "OK" : "FAIL");
    }

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(d_flag));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// WHILE graph: device-side loop, runs exactly N times.
// ---------------------------------------------------------------------------
static void test_while() {
    print_header("CUDA Graph Conditional: WHILE");

    int N = 1000;
    int *d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));

    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    // Default value 1, applied at the start of every launch -> do-while-ish.
    cudaGraphConditionalHandle handle;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));

    // Conditional WHILE node (no dependencies for this simple graph).
    cudaGraphNodeParams c_params = {};
    c_params.type = cudaGraphNodeTypeConditional;
    c_params.conditional.handle = handle;
    c_params.conditional.type = cudaGraphCondTypeWhile;
    c_params.conditional.size = 1;

    cudaGraphNode_t condNode;
    CUDA_CHECK(cudaGraphAddNode(&condNode, graph, nullptr, nullptr, 0, &c_params));

    cudaGraph_t bodyGraph = c_params.conditional.phGraph_out[0];

    void* body_args[3] = {&d_counter, &N, &handle};
    cudaGraphNodeParams body_params = {};
    body_params.type = cudaGraphNodeTypeKernel;
    body_params.kernel.func = (void*)while_body;
    body_params.kernel.gridDim = dim3(1);
    body_params.kernel.blockDim = dim3(32);
    body_params.kernel.kernelParams = body_args;

    cudaGraphNode_t bodyNode;
    CUDA_CHECK(cudaGraphAddNode(&bodyNode, bodyGraph, nullptr, nullptr, 0, &body_params));

    cudaGraphExec_t exec;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // Run the whole graph repeatedly, resetting the counter each time.
    bool ok = true;
    for (int rep = 0; rep < 3; rep++) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice));

        GpuTimer t;
        t.start();
        CUDA_CHECK(cudaGraphLaunch(exec, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        t.stop();

        int counter;
        CUDA_CHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
        ok &= (counter == N);
        printf("  rep %d: counter=%d (expected %d) time=%.4f ms %s\n", rep, counter, N, t.milliseconds(),
               (counter == N) ? "OK" : "FAIL");
    }

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(d_counter));

    if (!ok) {
        printf("  WHILE test FAILED\n");
        exit(1);
    }
}

int main() {
    print_header("CUDA Graph Conditional Nodes — GB10");
    test_if_else();
    test_while();
    return 0;
}
