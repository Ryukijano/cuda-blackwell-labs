# Project 12: Conditional CUDA Graphs — Analysis

## Results

### IF/ELSE
- `flag=1` correctly selects the true branch and writes `10`.
- `flag=0` correctly selects the false branch and writes `20`.

### WHILE
- Running a 1000-iteration device-side while loop takes ~3.28 ms for a single block.
- Each iteration launches the body graph once; the condition is evaluated on the GPU.

## Takeaways

1. `cudaGraphConditionalHandle` is the handle passed from host graph to device kernel.
2. `cudaGraphSetConditional` must be called by one lane/warp to avoid races.
3. `cudaGraphCondAssignDefault` sets the initial handle value at each graph launch.
4. Conditional nodes are not free: the WHILE loop overhead is visible (~3.3 µs per iteration). They pay off when the alternative is repeated CPU launches and synchronization.

## Potential next steps

- Use a larger body kernel per iteration so the overhead is amortized.
- Nest `IF` inside a `WHILE` for adaptive solvers.
- Combine with CUDA Graph capture for complex captured control flow.
