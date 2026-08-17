# Makefile for Project 2 — Memory Bandwidth & Latency Lab

NVCC ?= nvcc
ARCH ?= sm_121
SRC := bandwidth_test.cu
BIN := bandwidth_test
COMMON := ../../common

# Legacy simple benchmark still lives here for reference.
SIMPLE := bandwidth_simple
DEEP := deep_dive

.PHONY: all build simple deep test run workset stride block alloc uma ncu plot verify clean profile

all: build simple deep

# Full sweep harness from the earlier project-2 implementation
build: $(BIN)

$(BIN): $(SRC)
	$(NVCC) -arch=$(ARCH) -lineinfo -o $(BIN) $(SRC)

# Legacy simple benchmark
simple: $(SIMPLE)

$(SIMPLE): bandwidth_simple.cu
	$(NVCC) -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON) $< -o $@

# Deep-dive companion

$(DEEP): deep_dive.cu
	$(NVCC) -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON) $< -o $@

deep: $(DEEP)
	./$(DEEP) | tee results/deep_dive.txt

ptx: bandwidth_simple.cu
	$(NVCC) -arch=compute_121 -ptx -O2 -std=c++17 -I$(COMMON) $< -o bandwidth.ptx

profile: $(BIN)
	-ncu --set full --target-processes all -o results/02_memory_bandwidth.ncu-rep --force-overwrite ./$(BIN)

test: $(BIN)
	@mkdir -p results
	./$(BIN) --test

run: $(BIN)
	@mkdir -p results
	./$(BIN) --run
	python3 plot_results.py

workset: $(BIN)
	@mkdir -p results
	./$(BIN) --workset

stride: $(BIN)
	@mkdir -p results
	./$(BIN) --stride

block: $(BIN)
	@mkdir -p results
	./$(BIN) --block

alloc: $(BIN)
	@mkdir -p results
	./$(BIN) --alloc

uma: $(BIN)
	@mkdir -p results
	./$(BIN) --uma

ncu: $(BIN)
	@mkdir -p results
	-ncu --set full --target-processes all --csv --log-file results/ncu_sequential_read.csv ./$(BIN) --ncu sequential_read
	-ncu --set full --target-processes all --csv --log-file results/ncu_read_write_copy.csv ./$(BIN) --ncu read_write_copy
	-ncu --set full --target-processes all --csv --log-file results/ncu_saxpy.csv ./$(BIN) --ncu saxpy

plot: run
	python3 plot_results.py

verify: $(BIN)
	@mkdir -p results
	./$(BIN) --test
	@echo "Checking artifacts..."
	@for f in results/bandwidth_workset.csv results/bandwidth_stride.csv results/bandwidth_block.csv results/bandwidth_alloc.csv results/bandwidth_uma.csv; do \
	  if [ -f $$f ]; then echo "  $$f OK"; else echo "  $$f MISSING"; exit 1; fi; \
	done

clean:
	rm -f $(BIN) $(SIMPLE) $(DEEP) *.o *.ptx
	rm -f results/*.csv results/*.png results/*.txt results/*.log
