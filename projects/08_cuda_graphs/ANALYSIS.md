# Project 08: CUDA Graphs — Analysis

## Results

| Implementation | Total (1000 iters) | Per-iteration | Speedup |
|----------------|---------------------|---------------|---------|
| Eager stream (20 kernels) | 83.7 ms | 0.084 ms | 1.0x |
| CUDA Graph (20 kernels) | 44.9 ms | 0.045 ms | **1.86x** |
| Graph + H2D/kernel/D2H | 53.2 ms | 0.053 ms | 1.57x |
| Graph + dynamic params | 0.42 ms (100 iters) | 0.004 ms | - |

## Key Findings

1. **CUDA Graphs provide 1.86x speedup** for a sequence of 20 tiny kernels.
   - Eager: 83.7 ms total
   - Graph: 44.9 ms total
   - The savings come from reduced CPU launch overhead — the GPU does the same
     work, but the CPU issues one `cudaGraphLaunch` instead of 20 kernel launches.

2. **Per-iteration time drops from 0.084 ms to 0.045 ms.**
   - That means the CPU launch overhead per iteration is ~0.039 ms.
   - For 20 kernels, the overhead per kernel in eager mode is ~0.002 ms (2 μs).

3. **Graph with copies is slightly slower** (53.2 ms) than the pure graph (44.9 ms)
   because the H2D/D2H copies add memory-bound time. But it's still faster than
   the equivalent eager version would be because the copy+compute+ copy sequence
   is captured as one launch.

4. **Dynamic parameters work but have limitations.**
   - `cudaGraphExecKernelNodeSetParams` lets you change kernel parameters
     between launches, but it adds ~0.004 ms per iteration.
   - This is useful when the graph topology is fixed but a scalar/offset changes.
   - It does NOT support changing grid/block dimensions or function pointers.

## What CUDA Graphs Do

- **Capture once**: Use `cudaStreamBeginCapture` / `cudaStreamEndCapture` to
  record a sequence of operations into a `cudaGraph_t`.
- **Instantiate once**: `cudaGraphInstantiate` converts the graph into an
  executable `cudaGraphExec_t`.
- **Launch repeatedly**: `cudaGraphLaunch` submits the entire graph with one
  CPU call.

## Why It Helps on GB10

- The kernels are tiny (1MB, 256K elements each), so execution time is low.
- For tiny kernels, **CPU launch overhead dominates** the per-kernel time.
- CUDA Graphs pay the launch overhead once during capture/instantiation, then
  each graph launch is a single host call.

## Constraints and Pitfalls

1. **Static shapes only**: Kernel grid/block dimensions and argument counts must
   be the same for every launch.
2. **No conditional logic**: You cannot put `if` statements or CPU-side loops
   inside a captured graph.
3. **No stream-ordered allocation inside a graph**: Use pre-allocated buffers.
4. **Capture must be valid**: All operations between BeginCapture and EndCapture
   must be on the same stream.
5. **Dynamic parameters require update API**: Use
   `cudaGraphExecKernelNodeSetParams` and accept the overhead.

## Recommendations

- Use CUDA Graphs for **inference serving** with fixed model architecture and
  batch size.
- Use CUDA Graphs for **training loops** where the same forward/backward/optimizer
  sequence repeats with the same shapes.
- Do NOT use CUDA Graphs for **dynamic shapes** or **conditional execution**.
- On GB10, the benefit is visible for sequences of many small kernels where
  CPU launch overhead would otherwise be a large fraction of total time.
