# Project 19: Key Findings

- `atom.global.gpu` and `atom.global.sys` are both valid PTX on Volta+.
- On GB10 both map to the same physical LPDDR5X; the `sys` scope may still carry a small coherence-protocol overhead.
- The probe is cycle-accurate because it uses `clock64()` inside the kernel, not a host-side profiler.
