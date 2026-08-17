// endosight_kernel.cu
// RGB-validity point cloud filter with CUB stream compaction.

#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Kernel to compute flags: 1 if color sum > threshold, else 0
__global__ void rgb_validity_flag_kernel(
    const float* colors,
    int* flags,
    int n,
    float threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r = colors[3 * i + 0];
    float g = colors[3 * i + 1];
    float b = colors[3 * i + 2];
    flags[i] = (r + g + b > threshold) ? 1 : 0;
}

// Gather points and colors by index
__global__ void gather_points_colors_kernel(
    const float* points,
    const float* colors,
    const int* indices,
    float* out_points,
    float* out_colors,
    int m
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    int i = indices[j];
    out_points[3 * j + 0] = points[3 * i + 0];
    out_points[3 * j + 1] = points[3 * i + 1];
    out_points[3 * j + 2] = points[3 * i + 2];
    out_colors[3 * j + 0] = colors[3 * i + 0];
    out_colors[3 * j + 1] = colors[3 * i + 1];
    out_colors[3 * j + 2] = colors[3 * i + 2];
}

// Compact points and colors according to flags
std::vector<torch::Tensor> filter_rgb_valid_cuda(
    torch::Tensor points,
    torch::Tensor colors,
    float threshold
) {
    int n = points.size(0);
    TORCH_CHECK(points.is_cuda(), "points must be CUDA");
    TORCH_CHECK(colors.is_cuda(), "colors must be CUDA");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be Nx3");
    TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3, "colors must be Nx3");
    TORCH_CHECK(points.size(0) == colors.size(0), "points and colors must have same N");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Compute flags
    auto flags = torch::empty({n}, torch::dtype(torch::kInt32).device(points.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    rgb_validity_flag_kernel<<<blocks, threads, 0, stream>>>(
        colors.data_ptr<float>(),
        flags.data_ptr<int>(),
        n,
        threshold
    );

    // Output buffers (max size = n)
    auto out_points = torch::empty_like(points);
    auto out_colors = torch::empty_like(colors);
    auto out_count = torch::empty({1}, torch::dtype(torch::kInt32).device(points.device()));

    // Temp storage for CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int* d_flags = flags.data_ptr<int>();
    int* d_count = out_count.data_ptr<int>();

    // Allocate index array
    auto in_indices = torch::arange(n, torch::dtype(torch::kInt32).device(points.device()));
    auto out_indices = torch::empty({n}, torch::dtype(torch::kInt32).device(points.device()));

    // First CUB select on indices
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        in_indices.data_ptr<int>(), d_flags,
        out_indices.data_ptr<int>(), d_count, n, stream
    );
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        in_indices.data_ptr<int>(), d_flags,
        out_indices.data_ptr<int>(), d_count, n, stream
    );

    // Get selected count
    int selected = out_count.item<int>();

    // Slice outputs to actual size
    out_points = out_points.index({torch::indexing::Slice(0, selected)});
    out_colors = out_colors.index({torch::indexing::Slice(0, selected)});

    // Custom gather kernel: write points[ idx[j] ] and colors[ idx[j] ]
    // This avoids PyTorch index_select overhead and fuses point+color gather
    auto idx_slice = out_indices.slice(0, 0, selected);
    threads = 256;
    blocks = (selected + threads - 1) / threads;
    gather_points_colors_kernel<<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(),
        colors.data_ptr<float>(),
        idx_slice.data_ptr<int>(),
        out_points.data_ptr<float>(),
        out_colors.data_ptr<float>(),
        selected
    );

    cudaFree(d_temp_storage);

    return {out_points, out_colors};
}
