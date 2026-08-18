# Project 14 — Tiny Transformer Training

A minimal character-level GPT-like model in PyTorch, trained on the GB10.

## Run

```bash
cd projects/14_tiny_transformer
make run
```

This requires the `3d_recon` conda environment (or any environment with `torch` built for CUDA 13).

## Model

- 4 transformer blocks
- 4 attention heads, 384 embedding dims
- 128-token context length
- ~7.3 M parameters

## What it does

- Generates a synthetic byte-level dataset from a repeated phrase.
- Runs a short warmup + 500 training steps.
- Reports tokens/sec and an estimated TFLOPS count.

## Notes

- This is intentionally small so it completes in seconds.
- The FLOP estimate is a rough forward+backward count; actual measured throughput is dominated by kernel launch and Python overheads.
