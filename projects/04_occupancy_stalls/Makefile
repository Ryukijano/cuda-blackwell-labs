# Project 04: Occupancy & Stall Experiments — Makefile

PROJECT   := 04_occupancy_stalls
TARGET    := occupancy
SRCS      := occupancy.cu

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_121
COMMON    := ../../common
NVCCFLAGS := -arch=$(ARCH) -O2 -lineinfo -std=c++17 -I$(COMMON)

.PHONY: all clean run ptx profile stall_metrics

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	@mkdir -p ../../results
	./$(TARGET) | tee ../../results/$(PROJECT).txt

ptx: $(SRCS)
	$(NVCC) -arch=compute_121 -ptx -O2 -std=c++17 -I$(COMMON) $< -o occupancy.ptx

# Full Nsight Compute profile
profile: $(TARGET)
	$(CUDA_HOME)/bin/ncu --set full --target-processes all \
		-o ../../results/$(PROJECT).ncu-rep --force-overwrite ./$(TARGET)

# Specific stall metrics
stall_metrics: $(TARGET)
	$(CUDA_HOME)/bin/ncu --target-processes all \
		--metrics sm__warps_active.avg.per_cycle_active,\
smsp__warps_eligible.avg.per_cycle_active,\
smsp__inst_executed.avg.per_cycle_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_misc_per_issue_active.ratio \
		--csv ./$(TARGET) 2>&1 | tee ../../results/$(PROJECT)_stalls.csv

clean:
	rm -f $(TARGET) *.o *.ptx
