NVCC    := /usr/local/cuda/bin/nvcc
ARCH    := sm_121
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I../../common -Xcompiler -pthread
LDFLAGS := -lcudart -lpthread

TARGET  := psi_monitor
SRC     := psi_monitor.cu
SIZE_MB ?= 256
ITERS   ?= 50

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(SIZE_MB) $(ITERS)

clean:
	rm -f $(TARGET) *.o *.ptx
