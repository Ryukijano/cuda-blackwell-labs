# Project 06: Precision Lab — Makefile

PROJECT   := 06_precision_lab
TARGET    := precision_test
SRCS      := precision.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON)
LDFLAGS   := -lcublas -lcublasLt

.PHONY: all clean run ptx profile

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) | tee ../../results/$(PROJECT).txt

ptx: $(SRCS)
	$(NVCC) -arch=compute_121 -ptx -O2 -std=c++17 -I$(COMMON) $< -o precision.ptx

profile: $(TARGET)
	$(CUDA_HOME)/bin/ncu --set full --target-processes all \
		-o ../../results/$(PROJECT).ncu-rep --force-overwrite ./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.ptx
