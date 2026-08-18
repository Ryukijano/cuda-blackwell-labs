# Project 13: CUPTI Activity Trace — Makefile

PROJECT   := 13_cupti_trace
TARGET    := cupti_trace
SRCS      := cupti_trace.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON) -I$(CUDA_HOME)/include
LDFLAGS   := -lcudart -lcuda -lcupti

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	@mkdir -p ../../results
	LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH ./$(TARGET) | tee ../../results/$(PROJECT).txt

clean:
	rm -f $(TARGET) *.o *.ptx
