# Project 01: GB10 Hardware Probe — Makefile
# Builds the hardware probe that interrogates the GB10 DGX Spark.

PROJECT   := 01_hardware_probe
TARGET    := hardware_probe
SRCS      := probe.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON)
LDFLAGS   :=

OBJS := $(SRCS:.cu=.o)

.PHONY: all clean run ptx

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	@mkdir -p ../../results
	./$(TARGET) | tee ../../results/$(PROJECT).txt

# Also generate PTX for inspection (uses compute_121 for forward-compatible PTX)
ptx: probe.cu
	$(NVCC) -arch=compute_121 -ptx -O2 -std=c++17 -I$(COMMON) $< -o probe.ptx

clean:
	rm -f $(TARGET) *.o *.ptx
