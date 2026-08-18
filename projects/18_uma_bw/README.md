# Project 18: UMA PTX Bandwidth Probe

Measures true LPDDR5X bandwidth on the GB10 using PTX cache operators.

| Operator | Meaning |
|----------|---------|
| `ld.global.cg.f32` | Read that caches at L2 only, bypassing L1 |
| `st.global.cs.f32` | Streaming write that bypasses L2, forcing DRAM |

## Build and run

```bash
make
make run
make run SIZE_MB=256
```

The default buffer is 256 MB.
