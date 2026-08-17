# CUDA Blackwell Labs — Top-Level Makefile
# GB10 DGX Spark (SM121, CUDA 13.0, unified LPDDR5X)
#
# Usage:
#   make           — build all projects
#   make project01 — build only project 01 (hardware probe)
#   make clean     — remove all build artifacts
#   make test      — run all projects and save output to results/
#   make profile   — run nsys on all projects (where applicable)

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17
LDFLAGS   := -lcudart -lstdc++

PROJECTS := $(patsubst %/,%,$(dir $(wildcard projects/*/Makefile)))

.PHONY: all clean test profile $(PROJECTS)

all: $(PROJECTS)

$(PROJECTS):
	$(MAKE) -C $@ NVCC=$(NVCC) ARCH=$(ARCH)

clean:
	@for proj in $(PROJECTS); do \
		$(MAKE) -C $$proj clean 2>/dev/null || true; \
	done
	rm -rf results/

test: all
	@mkdir -p results
	@for proj in $(PROJECTS); do \
		echo "=== Running $$proj ==="; \
		$(MAKE) -C $$proj run; \
		echo ""; \
	done

profile: all
	@mkdir -p results
	@for proj in $(PROJECTS); do \
		echo "=== Profiling $$proj ==="; \
		$(CUDA_HOME)/bin/nsys profile -o results/$$(basename $$proj) --force-overwrite=true --stats=true ./$$proj/$$(basename $$proj) 2>&1 | tail -20; \
		echo ""; \
	done
