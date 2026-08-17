# Project 01 — GB10 Hardware Probe

Interrogates the NVIDIA DGX Spark (GB10, SM 12.1) and cross-checks CUDA-reported device properties with Linux `/proc/meminfo`.

## Build & run

```bash
cd projects/01_hardware_probe
make all      # build hardware_probe
make run      # print device props, memory cross-check, key takeaways
make ptx      # generate probe.ptx
make clean
```

## What it prints

- `cudaGetDeviceProperties` and `cudaDeviceGetAttribute` summary
- CUDA-reported free/total memory vs Linux-reported `/proc/meminfo`
- Compilation info (CUDA/driver/runtime versions)
- Key takeaways for GB10 programming

## Key takeaways

- GB10 is SM 12.1, not datacenter B200. No TMEM, WGMMA, or DSMEM.
- 128 GB of unified LPDDR5X, not HBM. Peak bandwidth is ~273 GB/s.
- `cudaMemGetInfo()` underreports on UMA; always cross-check with `/proc/meminfo`.
- Compile with `-arch=sm_121 -lineinfo` for this GPU.
