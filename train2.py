import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLOCK_SIZE = 1024          # context length (tokens per training window)
BATCH_SIZE = 8             # sequences per step
GRAD_ACCUM = 1             # raise this for a larger effective batch on limited VRAM
MAX_STEPS = 5000           # one "step" = one optimizer update
EVAL_INTERVAL = 500
EVAL_ITERS = 100
GRAD_CLIP = 1.0

LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        # [B, L, 3, n_heads, d_k] -> [3, B, n_heads, L, d_k]
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # each [B, n_heads, L, d_k]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)   # [B, n_heads, L, L]

        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v                               # [B, n_heads, L, d_k]
        out = out.transpose(1, 2).reshape(B, L, D)   # concat heads -> [B, L, D]
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)

        hidden_dim = int(d_model * 3 / 2)
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)   # gate
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)   # linear
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)   # back-projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        gate = F.silu(self.w1(h))
        linear = self.w2(h)
        x = x + self.dropout(self.w3(gate * linear))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, layers=12, max_len=BLOCK_SIZE):
        super().__init__()
        self.max_len = max_len
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Parameter(torch.zeros(max_len, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(layers)])
        self.lnF = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Optional weight tying:
        # self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor([idx], dtype=torch.long, device=self.wpe.device)
        B, T = idx.shape
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"
        x = self.tok_emb(idx) + self.wpe[:T]       # [B, T, d_model]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.lnF(x))           # [B, T, vocab]

    @torch.no_grad()
    def generate(self, prompt, max_new=50, temperature=1.0):
        self.eval()
        device = self.wpe.device
        idx = torch.tensor([self.tok.encode(prompt)], dtype=torch.long, device=device)
        for _ in range(max_new):
            logits = self.forward(idx[:, -self.max_len:])   # feed full (cropped) context
            logits = logits[:, -1, :] / temperature
            p = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(p, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return self.tok.decode(idx[0].tolist())


# ----------------------------------------------------------------------------
# Data: tokenize once into a flat uint16 stream, cache to disk, sample windows
# ----------------------------------------------------------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
EOT = tokenizer.eos_token_id   # 50256, <|endoftext|> — used as a document separator


def prepare_data(split, cache_path=None):
    """Tokenize a WikiText-103 split into one contiguous uint16 array on disk."""
    cache_path = cache_path or f"wikitext103_{split}.bin"
    if os.path.exists(cache_path):
        data = np.memmap(cache_path, dtype=np.uint16, mode="r")
        print(f"[{split}] loaded {len(data):,} cached tokens from {cache_path}")
        return data

    print(f"[{split}] tokenizing -> {cache_path}")
    ds = load_dataset("wikitext", "wikitext-103-v1", split=split)

    def tok_fn(examples):
        out_ids, out_len = [], []
        for text in examples["text"]:
            ids = tokenizer.encode(text)
            if ids:
                ids.append(EOT)            # separate documents
            out_ids.append(ids)
            out_len.append(len(ids))
        return {"ids": out_ids, "len": out_len}

    tokenized = ds.map(
        tok_fn, batched=True, remove_columns=ds.column_names,
        num_proc=8, desc="tokenizing",
    )

    total = int(np.sum(tokenized["len"], dtype=np.int64))
    arr = np.memmap(cache_path, dtype=np.uint16, mode="w+", shape=(total,))
    offset = 0
    n_shards = math.ceil(len(tokenized) / 4096)
    for shard in tqdm(tokenized.iter(batch_size=4096), total=n_shards, desc="writing"):
        block = np.concatenate([np.asarray(s, dtype=np.uint16) for s in shard["ids"] if s])
        arr[offset: offset + len(block)] = block
        offset += len(block)
    arr.flush()
    print(f"[{split}] wrote {total:,} tokens")
    return np.memmap(cache_path, dtype=np.uint16, mode="r")


def get_batch(data, block_size, batch_size, device):
    """Sample `batch_size` random windows of length block_size (+1 for the shift)."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            logits = model(x)
            losses[k] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item()
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(step):
    """Linear warmup then cosine decay to MIN_LR."""
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(model, train_data, val_data):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )
    splits = {"train": train_data, "val": val_data}
    model.train()

    pbar = tqdm(range(MAX_STEPS), desc="training")
    for step in pbar:
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        # gradient accumulation for a larger effective batch
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            (loss / GRAD_ACCUM).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.2e}")

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            stats = estimate_loss(model, splits, BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS)
            ppl = math.exp(stats["val"])
            print(
                f"\nstep {step}: train {stats['train']:.4f} | "
                f"val {stats['val']:.4f} | val ppl {ppl:.2f}"
            )


if __name__ == "__main__":
    train_data = prepare_data("train")
    val_data = prepare_data("validation")

    model = GPT2().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.1f}M | device: {DEVICE}")

    train(model, train_data, val_data)
    print("Generated:", model.generate("The future of AI is", max_new=40))
