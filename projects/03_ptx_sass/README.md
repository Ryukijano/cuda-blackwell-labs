# Project 03 — CUDA → PTX → SASS Pipeline

Trace CUDA source code through the full compilation pipeline and compare compiler optimizations, kernel variants, and GPU architectures.

## Quick start

```bash
cd projects/03_ptx_sass
make all         # build pipeline binary for SM121
make run         # benchmark all SAXPY variants on the GB10
make ptx         # generate PTX (pipeline.ptx)
make sass        # generate SASS for -Xptxas -O0 and -O3
make sass_all    # dump SASS for the default binary
make arch_compare # build and dump SASS for sm_80, sm_90, sm_121
make clean       # remove generated binaries and dumps
```

## Variants

| Kernel | Purpose |
|---|---|
| `saxpy_basic` | Baseline SAXPY |
| `saxpy_restrict` | `__restrict__` aliasing hint |
| `saxpy_vec4` | `float4` vectorized loads/stores |
| `saxpy_shared` | Shared memory tiling |
| `saxpy_launchbounds` | `__launch_bounds__(256, 4)` occupancy hint |
| `saxpy_unrolled` | Manual loop unrolling |
| `reduce_sum` | Reduction pattern for SASS comparison |

## Artifacts

- `pipeline` — benchmark executable (ignored by git, build with `make`)
- `pipeline.ptx` — generated PTX
- `sass_O0.txt`, `sass_O3.txt` — SASS dumps for O0 vs O3
- `sass_sm80.txt`, `sass_sm90.txt`, `sass_sm121.txt` — cross-architecture SASS
- `../../results/03_ptx_sass.txt` — runtime results

The detailed annotated analysis is in `ANALYSIS.md`.
