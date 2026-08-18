# Project 11 — Tensor Memory Accelerator (TMA)

Demonstrates 2D descriptor-driven tile copies using the TMA hardware engine on the GB10 (SM121).

## Build & run

```bash
cd projects/11_tma_lab
make all
make run
make ptx      # generate tma.ptx
make clean
```

## What it does

- Builds a `CUtensorMap` on the host with `cuTensorMapEncodeTiled`.
- Each CTA issues one `cp.async.bulk.tensor.2d` TMA load to bring a 64×64 tile from global to shared memory.
- Threads perform a tiny in-place transform (`+1`) to prove the data is resident.
- Issues a `cp.async.bulk.tensor.2d.global.shared` TMA store to write the tile back to global memory.
- Compares against a naive per-thread copy of the same 64×64 tiles.

## Key PTX instructions used

- `mbarrier.init.shared.b64` — initialize a shared mbarrier.
- `mbarrier.arrive.expect_tx.release.cta` — tell the barrier how many bytes the TMA will deliver.
- `cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes` — TMA global → shared load.
- `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group` — TMA shared → global store.
- `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group` — wait for the store to drain.

## Why?

TMA lets the hardware walk the tensor layout using a descriptor instead of requiring every thread to compute global addresses. On 2D/3D tile workloads it reduces instruction overhead and can avoid bank conflicts with swizzled shared layouts. This minimal example shows the API; peak performance needs larger tiles, swizzling, and pipelined staging.
