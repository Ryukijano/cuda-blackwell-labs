# Project 03: CUDA → PTX → SASS Pipeline — Makefile

PROJECT   := 03_ptx_sass
TARGET    := pipeline
SRCS      := pipeline.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -lineinfo -std=c++17 -I$(COMMON)

.PHONY: all clean run ptx sass sass_all arch_compare

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -O2 $< -o $@

run: $(TARGET)
	@mkdir -p ../../results
	./$(TARGET) | tee ../../results/$(PROJECT).txt

# Generate PTX (virtual ISA, forward-compatible)
ptx: $(SRCS)
	$(NVCC) -arch=compute_121 -ptx -O2 -std=c++17 -I$(COMMON) $< -o pipeline.ptx
	@echo "=== PTX generated: pipeline.ptx ==="
	@echo "Inspect with: cat pipeline.ptx"

# Generate SASS for O0 and O3 (use -Xptxas for device code optimization)
sass: $(TARGET)
	$(NVCC) $(NVCCFLAGS) -Xptxas -O0 $(SRCS) -o pipeline_O0
	$(NVCC) $(NVCCFLAGS) -Xptxas -O3 $(SRCS) -o pipeline_O3
	$(CUDA_HOME)/bin/cuobjdump --dump-sass pipeline_O0 > sass_O0.txt
	$(CUDA_HOME)/bin/cuobjdump --dump-sass pipeline_O3 > sass_O3.txt
	@echo "=== SASS generated: sass_O0.txt, sass_O3.txt ==="
	@echo "Compare: diff sass_O0.txt sass_O3.txt"

# Generate SASS for all variants
sass_all: sass
	$(CUDA_HOME)/bin/cuobjdump --dump-sass $(TARGET) > sass_all.txt
	@echo "=== All SASS in sass_all.txt ==="

# Compare SASS across architectures
arch_compare: $(SRCS)
	$(NVCC) -arch=sm_80  -lineinfo -O2 -std=c++17 -I$(COMMON) $(SRCS) -o pipeline_sm80
	$(NVCC) -arch=sm_90  -lineinfo -O2 -std=c++17 -I$(COMMON) $(SRCS) -o pipeline_sm90
	$(NVCC) -arch=sm_121 -lineinfo -O2 -std=c++17 -I$(COMMON) $(SRCS) -o pipeline_sm121
	$(CUDA_HOME)/bin/cuobjdump --dump-sass pipeline_sm80  > sass_sm80.txt
	$(CUDA_HOME)/bin/cuobjdump --dump-sass pipeline_sm90  > sass_sm90.txt
	$(CUDA_HOME)/bin/cuobjdump --dump-sass pipeline_sm121 > sass_sm121.txt
	@echo "=== Architecture comparison SASS generated ==="
	@echo "  sass_sm80.txt  (Ampere)"
	@echo "  sass_sm90.txt  (Hopper)"
	@echo "  sass_sm121.txt (Blackwell GB10)"

clean:
	rm -f $(TARGET) pipeline_O0 pipeline_O3 pipeline_sm80 pipeline_sm90 pipeline_sm121
	rm -f *.ptx *.txt *.o
