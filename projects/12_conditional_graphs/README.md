# Project 12 — CUDA Graph Conditional Nodes

Demonstrates dynamic control flow inside a CUDA graph using conditional `IF/ELSE` and `WHILE` nodes.

## Build & run

```bash
cd projects/12_conditional_graphs
make all
make run
make clean
```

## What it does

### 1. `IF/ELSE` graph
- A kernel reads a device flag and calls `cudaGraphSetConditional(handle, flag)`.
- The conditional node routes to one of two child graphs:
  - `true` branch writes `10` to a device value.
  - `false` branch writes `20`.
- The host then runs the graph with `flag=1` and `flag=0` and verifies the output.

### 2. `WHILE` graph
- A conditional `WHILE` node with default value `1` acts as a do-while loop.
- The body kernel increments a device counter and calls `cudaGraphSetConditional(handle, continue?)`.
- The loop runs exactly `N` times before the condition becomes `0`.

## Key APIs

- `cudaGraphConditionalHandleCreate`
- `cudaGraphAddNode` with `cudaGraphNodeTypeConditional`
- `cudaGraphSetConditional` from inside a kernel
- `cudaGraphCondTypeIf`, `cudaGraphCondTypeWhile`, `cudaGraphCondAssignDefault`

## Why?

Conditional graphs let the GPU make runtime control-flow decisions without returning to the CPU. This is useful for iterative solvers, data-dependent dispatch, and dynamic loops.
