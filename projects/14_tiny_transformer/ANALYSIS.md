# Project 14: Tiny Transformer Training — Analysis

## Results

| Metric | Value |
|--------|-------|
| Parameters | 7.34 M |
| Sequence length | 128 |
| Batch size | 64 |
| Train steps | 500 |
| Time | ~30 s |
| Avg loss | 0.0149 |
| Tokens/sec | ~139,000 |
| Est. TFLOPS | ~4.3 |

## Observations

- The model converges quickly on the tiny synthetic dataset (repeated phrase).
- Throughput is limited by the small batch and short sequence, not by the GPU's peak compute.
- The estimated TFLOPS is a fraction of the GB10 peak because the model is tiny and the workload is memory- and Python-overhead-bound.

## Takeaways

1. GB10 can train a 7M parameter decoder in PyTorch without any special flags.
2. For larger models, `torch.compile`, `torch.cuda.amp` (or BF16), and larger effective batch sizes would push the hardware harder.
3. A more interesting next step would be to train on a real dataset and compare BF16 vs FP8/NVFP4 in `transformerengine` or a custom mixed-precision kernel.
