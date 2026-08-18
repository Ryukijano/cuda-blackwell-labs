# Project 20: Key Findings

- `ld.global.cv` bypasses L1/L2 and forces a real memory access, so the latency includes any UMA fault/migration cost.
- COLD pass shows the cost when the OS first resolves the page for the GPU.
- WARM pass shows the steady-state resident-on-GPU latency.
- On hardware-coherent UMA the cold/warm ratio may be small because the physical memory is already allocated; the difference is mostly page-table / TLB resolution.
