// Project 13: CUPTI Activity Trace
//
// A minimal CUPTI activity-trace program that records kernel launches,
// timings, grid/block sizes, and memory copies without needing root.

#include "cuda_utils.h"

#ifndef CUPTI_API_CALL
#define CUPTI_API_CALL(apiFunctionCall)                                                \
do {                                                                                   \
    CUptiResult _status = apiFunctionCall;                                             \
    if (_status != CUPTI_SUCCESS) {                                                    \
        const char* _errstr;                                                           \
        cuptiGetResultString(_status, &_errstr);                                       \
        fprintf(stderr, "CUPTI error at %s:%d: %s\n", __FILE__, __LINE__, _errstr);    \
        exit(1);                                                                       \
    }                                                                                  \
} while (0)
#endif
#include <cupti.h>
#include <cupti_activity.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static size_t s_numRecords = 0;

// ---------------------------------------------------------------------------
// CUPTI buffer callbacks
// ---------------------------------------------------------------------------
static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *size = 8 * 1024 * 1024;  // 8 MB
    *buffer = (uint8_t*)malloc(*size);
    *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t *buffer, size_t, size_t validSize) {
    if (validSize == 0) {
        free(buffer);
        return;
    }

    CUpti_Activity *record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                auto *k = (CUpti_ActivityKernel10*)record;
                if (k->start == 0 && k->end == 0) break;
                double us = (double)(k->end - k->start) / 1000.0;
                printf("  %-50s %7.2f us  grid=(%d,%d,%d) block=(%d,%d,%d)\n",
                       k->name ? k->name : "?",
                       us,
                       k->gridX, k->gridY, k->gridZ,
                       k->blockX, k->blockY, k->blockZ);
                s_numRecords++;
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto *m = (CUpti_ActivityMemcpy6*)record;
                printf("  [memcpy] %7.2f us  bytes=%llu  kind=%u\n",
                       (double)(m->end - m->start) / 1000.0,
                       (unsigned long long)m->bytes,
                       (unsigned)m->copyKind);
                s_numRecords++;
                break;
            }
            default:
                break;
        }
    }
    free(buffer);
}

// ---------------------------------------------------------------------------
// Sample workload
// ---------------------------------------------------------------------------
__global__ void warmup_kernel(int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = i * 2;
}

int main() {
    print_header("CUPTI Activity Trace — GB10");

    // Register CUPTI callbacks and enable the activity kinds we want.
    CUPTI_API_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));

    int N = 1 << 20;
    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

    int blocks = (N + 255) / 256;

    // Warmup to avoid first-launch bookkeeping.
    warmup_kernel<<<blocks, 256>>>(d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch a few kernels and let CUPTI record them.
    printf("\nRecorded kernels and copies:\n");
    for (int i = 0; i < 3; i++) {
        warmup_kernel<<<blocks, 256>>>(d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Flush and disable.
    CUPTI_API_CALL(cuptiActivityFlushAll(1));
    CUPTI_API_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_API_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_API_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));

    printf("\n  Total activity records: %zu\n", s_numRecords);

    cudaFree(d_out);
    return 0;
}
