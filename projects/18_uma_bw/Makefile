NVCC    := /usr/local/cuda/bin/nvcc
ARCH    := sm_121
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I../../common
LDFLAGS := -lcudart

TARGET  := uma_bw
SRC     := uma_bw.cu
SIZE_MB ?= 256

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(SIZE_MB)

clean:
	rm -f $(TARGET) *.o *.ptx
