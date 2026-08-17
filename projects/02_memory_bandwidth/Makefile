# Project 02: Memory Bandwidth & Latency Lab — Makefile

PROJECT   := 02_memory_bandwidth
TARGET    := bandwidth_test
DEEP      := deep_dive
SRCS      := bandwidth.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON)

.PHONY: all clean run deep ptx profile

all: $(TARGET) $(DEEP)

$(TARGET): bandwidth.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(DEEP): deep_dive.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) | tee ../../results/$(PROJECT).txt

deep: $(DEEP)
	./$(DEEP) | tee ../../results/$(PROJECT)_deep.txt

ptx: bandwidth.cu
	$(NVCC) -arch=compute_121 -ptx -O2 -std=c++17 -I$(COMMON) $< -o bandwidth.ptx

profile: $(TARGET)
	$(CUDA_HOME)/bin/ncu --set full --target-processes all -o ../../results/$(PROJECT).ncu-rep --force-overwrite ./$(TARGET)

clean:
	rm -f $(TARGET) $(DEEP) *.o *.ptx
