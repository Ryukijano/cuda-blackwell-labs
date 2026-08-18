NVCC    := /usr/local/cuda/bin/nvcc
ARCH    := sm_121
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I../../common
LDFLAGS := -lcudart

TARGET  := uma_fault
SRC     := uma_fault.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.ptx
