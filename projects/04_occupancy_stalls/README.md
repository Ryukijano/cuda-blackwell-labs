# Project 04 — Occupancy & Stall Experiments

Systematically vary register pressure, shared memory, block size, and warp divergence to see what limits performance on the GB10 SM121.

## Quick start

```bash
cd projects/04_occupancy_stalls
make all         # build occupancy
make run         # run all experiments
make ptx         # generate PTX
make profile     # full Nsight Compute profile (requires perf counter permissions)
make stall_metrics # specific stall metrics
make clean       # remove generated binaries
```

## Experiments

| Part | What it varies | Key takeaway |
|---|---|---|
| A | Register pressure (`float regs[N_REGS]`) | The compiler reuses registers aggressively; actual count is ~11 for all N. |
| B | Shared memory (1–96 KB) | Occupancy cliff at 64 KB; <48 KB is usually fine. |
| C | Block size (32–1024) | 64–512 threads/block is optimal on GB10. |
| D | Warp divergence | Minimal effect for memory-bound kernels. |
| E | Instruction dependencies | ILP helps little; memory latency dominates. |
| F | High occupancy vs fast | Access pattern matters more than occupancy. |

## Artifacts

- `occupancy` — benchmark executable (ignored by git)
- `occupancy.ptx` — generated PTX
- `../../results/04_occupancy_stalls.txt` — runtime output
- `ANALYSIS.md` — annotated findings and tables
