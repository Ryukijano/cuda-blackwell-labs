// endosight_kernel.cpp
// PyTorch binding for Endosight custom CUDA kernel.

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> filter_rgb_valid_cuda(
    torch::Tensor points,
    torch::Tensor colors,
    float threshold
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_rgb_valid", &filter_rgb_valid_cuda, "Filter point cloud by RGB validity (CUDA)");
}
