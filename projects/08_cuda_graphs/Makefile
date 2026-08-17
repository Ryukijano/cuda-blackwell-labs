# Project 08: CUDA Graphs — Makefile

PROJECT   := 08_cuda_graphs
TARGET    := graph_test
SRCS      := graphs.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON)

.PHONY: all clean run ptx profile

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) | tee ../../results/$(PROJECT).txt

ptx: $(SRCS)
	$(NVCC) -arch=compute_121 -ptx -O2 -std=c++17 -I$(COMMON) $< -o graphs.ptx

profile: $(TARGET)
	$(CUDA_HOME)/bin/nsys profile --stats=true \
		-o ../../results/$(PROJECT).nsys-rep ./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.ptx
