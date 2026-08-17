# Project 08 — CUDA Graphs

Benchmarks CPU launch overhead vs CUDA Graphs for repeated small-kernel sequences on GB10.

## Build & run

```bash
cd projects/08_cuda_graphs
make all      # build graph_test
make run      # run eager, graph, graph+copies, and dynamic params benchmarks
make ptx      # generate graphs.ptx
make profile  # Nsight Systems profile
make clean
```

## Implementations

1. **Eager stream launch** — 20 tiny kernels per iteration, 1000 iterations.
2. **CUDA Graph capture** — same 20 kernels captured once, launched as a graph.
3. **Graph with H2D + kernel + D2H** — includes memory copies in the graph.
4. **Graph with dynamic kernel params** — topology fixed, scalar params updated per launch.

## Key findings

- CUDA Graphs give a **~1.9x speedup** for repeated tiny-kernel sequences.
- CPU launch overhead per eager kernel is ~2 µs; a graph launch replaces 20 launches with one.
- Graphs work best for **static shapes** and **fixed topology**.
- Dynamic parameters via `cudaGraphExecKernelNodeSetParams` add ~4 µs per update.
