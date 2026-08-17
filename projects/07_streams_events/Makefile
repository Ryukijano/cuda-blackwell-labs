# Project 07: Streams, Events, Async Allocation — Makefile

PROJECT   := 07_streams_events
TARGET    := streams_test
SRCS      := streams.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON)
# NVTX 3 is header-only in CUDA 13

.PHONY: all clean run ptx profile

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) | tee ../../results/$(PROJECT).txt

ptx: $(SRCS)
	$(NVCC) -arch=compute_121 -ptx -O2 -std=c++17 -I$(COMMON) $< -o streams.ptx

profile: $(TARGET)
	$(CUDA_HOME)/bin/nsys profile --stats=true \
		-o ../../results/$(PROJECT).nsys-rep ./$(TARGET)

profile: $(TARGET)
	$(CUDA_HOME)/bin/nsys profile --stats=true \
		-o ../../results/$(PROJECT).nsys-rep ./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.ptx
