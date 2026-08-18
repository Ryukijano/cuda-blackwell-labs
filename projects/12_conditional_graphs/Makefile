# Project 12: CUDA Graph Conditional Nodes — Makefile

PROJECT   := 12_conditional_graphs
TARGET    := cond_graphs
SRCS      := cond_graphs.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON)
LDFLAGS   := -lcudart

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	@mkdir -p ../../results
	./$(TARGET) | tee ../../results/$(PROJECT).txt

clean:
	rm -f $(TARGET) *.o *.ptx
