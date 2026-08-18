NVCC    := /usr/local/cuda/bin/nvcc
ARCH    := sm_121
CUTLASS := $(CURDIR)/../15_cutlass_gemm/cutlass

NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 --expt-relaxed-constexpr \
             -I../../common -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include
LDFLAGS := -lcudart

TARGET  := cute_gemm
SRC     := cute_gemm.cu
M       ?= 512
K       ?= $(M)

.PHONY: all fetch run clean

all: $(TARGET)

$(TARGET): $(SRC)
	@if [ ! -d "$(CUTLASS)/include/cute" ]; then \
	  echo "CUTLASS not found; fetching into projects/15_cutlass_gemm/cutlass ..."; \
	  $(MAKE) -C $(dir $(CUTLASS)) fetch; \
	fi
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(M) $(K)

clean:
	rm -f $(TARGET) *.o *.ptx
