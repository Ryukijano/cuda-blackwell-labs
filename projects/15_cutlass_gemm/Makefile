# Project 15: CUTLASS 3.8 Hello GEMM — Makefile

PROJECT   := 15_cutlass_gemm
TARGET    := cutlass_gemm
SRCS      := cutlass_gemm.cu

CUDA_HOME ?= /usr/local/cuda
CUTLASS   := $(CURDIR)/cutlass
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
INCLUDES  := -I$(COMMON) -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 --expt-relaxed-constexpr $(INCLUDES)
LDFLAGS   := -lcudart

.PHONY: all clean run fetch

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	@mkdir -p ../../results
	./$(TARGET) | tee ../../results/$(PROJECT).txt

fetch:
	@if [ ! -d "$(CUTLASS)" ]; then \
		git clone --depth 1 --branch v3.8.0 https://github.com/NVIDIA/cutlass.git $(CUTLASS); \
	fi

clean:
	rm -f $(TARGET) *.o *.ptx
