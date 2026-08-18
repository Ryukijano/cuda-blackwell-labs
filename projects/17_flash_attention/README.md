# Project 17 — FlashAttention-style Online Softmax Attention

A self-contained CUDA implementation of the online-softmax attention update that underlies FlashAttention.

## Build & run

```bash
cd projects/17_flash_attention
make all
make run
```

## What it does

- Computes `O = softmax(Q K^T / sqrt(D)) V` for one query row at a time.
- Avoids materialising the full `N x N` attention matrix.
- Maintains a running maximum `m` and running sum `l`; rescales the running output `o` at each key:

```
m_new = max(m, s)
l     = l * exp(m - m_new) + exp(s - m_new)
o     = o * exp(m - m_new) + exp(s - m_new) * V[j]
```

- After the last key, `O = o / l`.
- Result is compared against a naive CPU attention.

## Notes

- This is a **correctness-first, serial-per-query** probe. Each query is handled by a separate block with a single thread that serialises over keys.
- A production version would tile `K` and `V` through shared memory and parallelise the dot products and epilogue across warps/threads.
