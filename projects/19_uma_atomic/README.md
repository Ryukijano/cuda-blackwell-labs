# Project 19: UMA Atomic Coherence Probe

Measures cycle-accurate atomic latency on GB10 with two PTX scope variants:

| PTX | Scope |
|-----|-------|
| `atom.global.gpu.add.u32` | GPU scope (local to GPU complex) |
| `atom.global.sys.add.u32` | System scope (NVLink-C2C coherent) |

The `SYS/GPU` latency ratio is the coherence signal. On discrete GPUs it is ~1x; on hardware-coherent UMA it reveals the cost of the CPU-GPU coherence protocol.

## Build and run

```bash
make
make run
```
