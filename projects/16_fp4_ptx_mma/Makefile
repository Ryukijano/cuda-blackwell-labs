# Project 16: FP4 PTX MMA — Makefile

PROJECT   := 16_fp4_ptx_mma
TARGET    := fp4_mma
SRCS      := fp4_mma.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
VIRTUAL   := compute_121a
GPU       := sm_121a
COMMON    := ../../common
NVCCFLAGS := -arch=$(VIRTUAL) -code=$(GPU) -O2 -lineinfo -std=c++17 -I$(COMMON)
LDFLAGS   := -lcudart

.PHONY: all clean run ptx

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	@mkdir -p ../../results
	./$(TARGET) | tee ../../results/$(PROJECT).txt

ptx: $(SRCS)
	$(NVCC) -arch=compute_121a -ptx -O2 -std=c++17 -I$(COMMON) $< -o fp4_mma.ptx

clean:
	rm -f $(TARGET) *.o *.ptx
