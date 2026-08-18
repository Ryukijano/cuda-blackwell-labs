#!/usr/bin/env python
"""Project 14: Tiny Transformer Training on GB10

A minimal character-level language model in PyTorch, tuned for the GB10
(128 GB unified, ~213 FP16 TFLOPS).  The hyper-parameters are deliberately
small so the training loop fits in a few seconds on a single GPU.
"""

import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters tuned for GB10 (small model, small batch, short sequence)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
BLOCK_SIZE = 128
TRAIN_STEPS = 500
LEARNING_RATE = 3e-4
N_LAYER = 4
N_HEAD = 4
N_EMBD = 384
DROPOUT = 0.0
WARMUP = 50

VOCAB_SIZE = 256  # bytes

# ---------------------------------------------------------------------------
# Simple byte-level dataset: a repetitive synthetic text
# ---------------------------------------------------------------------------
def make_data(n=100_000):
    pattern = bytearray(b"the quick brown fox jumps over the lazy dog ")
    repeats = (n // len(pattern)) + 1
    return (pattern * repeats)[:n]

class ByteDataset:
    def __init__(self, data, block_size):
        self.data = torch.frombuffer(data, dtype=torch.uint8).long()
        self.block_size = block_size

    def get_batch(self, batch_size, device):
        ix = torch.randint(0, len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# Minimal GPT-like model
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, num_heads, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def estimate_flops_per_step(batch_size, block_size, n_embd, n_layer, n_head, vocab_size):
    """Very rough forward-pass FLOP count for a decoder-only Transformer."""
    # Embedding lookups are small.
    # Each attention head: QK^T (B * T * T * head_size) + softmax (ignored)
    #   + @V (B * T * T * head_size), per head * n_head => ~2 * B * T^2 * n_embd
    # Feed-forward: 2 * B * T * (4 * n_embd * n_embd)  + 2 * B * T * (n_embd * 4*n_embd) (two lin)
    # Actually: 2 * 4 * n_embd^2 * B * T for forward of ffn
    # LM head: 2 * B * T * n_embd * vocab_size
    n_params = (
        vocab_size * n_embd                        # token emb
        + block_size * n_embd                      # pos emb
        + n_layer * (4 * n_embd * n_embd + 4 * n_embd * n_embd)  # attn qkv+proj, ffn
        + n_embd * vocab_size                      # lm head
    )
    # A common rule: forward FLOPs ~ 2 * params per token, plus attention B*T^2*n_embd
    forward_flops = 2 * n_params * (batch_size * block_size)
    forward_flops += 2 * n_layer * batch_size * block_size * block_size * n_embd
    # Backward ~ 2x forward
    return 3 * forward_flops

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data = make_data()
    ds = ByteDataset(data, BLOCK_SIZE)
    model = TinyGPT(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params/1e6:.2f} M parameters")
    print(f"  layers={N_LAYER}, heads={N_HEAD}, embd={N_EMBD}, block={BLOCK_SIZE}, batch={BATCH_SIZE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))

    model.train()

    # Warmup
    for _ in range(WARMUP):
        x, y = ds.get_batch(BATCH_SIZE, device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    # Timed run
    t0 = time.time()
    total_loss = 0.0
    for step in range(TRAIN_STEPS):
        x, y = ds.get_batch(BATCH_SIZE, device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    avg_loss = total_loss / TRAIN_STEPS
    tokens = BATCH_SIZE * BLOCK_SIZE * TRAIN_STEPS
    tok_per_sec = tokens / elapsed
    flops_per_step = estimate_flops_per_step(BATCH_SIZE, BLOCK_SIZE, N_EMBD, N_LAYER, N_HEAD, VOCAB_SIZE)
    tfops = flops_per_step * TRAIN_STEPS / elapsed / 1e12

    print(f"\n--- Training results ---")
    print(f"  steps:            {TRAIN_STEPS}")
    print(f"  tokens:           {tokens}")
    print(f"  time:             {elapsed:.3f} s")
    print(f"  avg loss:         {avg_loss:.4f}")
    print(f"  tokens/sec:       {tok_per_sec:,.0f}")
    print(f"  est. TFLOPS:      {tfops:.2f}")

if __name__ == "__main__":
    main()
