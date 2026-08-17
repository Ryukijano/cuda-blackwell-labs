from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="endosight_kernel",
    ext_modules=[
        CUDAExtension(
            name="endosight_kernel",
            sources=[
                "endosight_binding.cpp",
                "endosight_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],
                "nvcc": ["-O2", "-arch=sm_121", "-std=c++17"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
