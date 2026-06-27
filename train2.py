import os
import math
import json
import time
import random
import datetime
import argparse
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
def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
AUTOCAST_DEVICE = DEVICE if DEVICE in ("cuda", "cpu", "mps") else "cpu"
DTYPE = "fp32"             # "fp32" or "bf16"; bf16 turns on autocast (no GradScaler needed)
USE_AMP = False            # derived from DTYPE in __main__

BLOCK_SIZE = 1024          # processed length (s-token positions per sequence)
BATCH_SIZE = 8
GRAD_ACCUM = 1
MAX_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 100
GRAD_CLIP = 1.0

LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01

# --- Token Superposition Training (TST) ---
# mode: "off" = plain next-token baseline
#       "full" = input + output superposition (s-fold data throughput, equal-FLOPs)
#       "output" = output-only superposition (no extra data; better for small corpora)
TST_MODE = "full"
TST_BAG_SIZE = 6           # superposition bag size s   (paper robust for s in [4, 8])
TST_RATIO = 0.3            # fraction of MAX_STEPS spent in superposition (paper [0.2, 0.4])

CKPT_DIR = "checkpoints"
CKPT_INTERVAL = EVAL_INTERVAL
RESUME = True
COMPILE = False            # torch.compile (most reliable on CUDA)
LOG_FILE = "results.jsonl" # one appended JSON record per completed run
RUN_NAME = None            # defaults to the checkpoint dir name


def amp_autocast():
    """bf16 autocast when DTYPE='bf16'; a no-op context otherwise. backward()
    must run OUTSIDE this context. bf16 needs no GradScaler (unlike fp16)."""
    return torch.autocast(device_type=AUTOCAST_DEVICE, dtype=torch.bfloat16, enabled=USE_AMP)


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
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
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
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
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
        # NB: embedding + lm_head are shared UNCHANGED across both TST phases.
        # That representation alignment is what the paper (§5.3) identifies as
        # the reason recovery works without an adapter step. Do not re-init them.

    def embed(self, idx):
        """Token embedding. Accepts [B, T] (standard) or [B, T, s] (superposed:
        average the s token embeddings in each bag, summed in fp32 for precision)."""
        if idx.dim() == 3:
            h = self.tok_emb(idx[..., 0]).float()
            for i in range(1, idx.size(-1)):
                h = h + self.tok_emb(idx[..., i]).float()
            return (h / idx.size(-1)).to(self.tok_emb.weight.dtype)
        return self.tok_emb(idx)

    def forward(self, idx):
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor([idx], dtype=torch.long, device=self.wpe.device)
        T = idx.shape[1]
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"
        x = self.embed(idx) + self.wpe[:T]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.lnF(x))

    @torch.no_grad()
    def generate(self, prompt, max_new=50, temperature=1.0):
        self.eval()
        device = self.wpe.device
        idx = torch.tensor([self.tok.encode(prompt)], dtype=torch.long, device=device)
        for _ in range(max_new):
            logits = self.forward(idx[:, -self.max_len:])
            logits = logits[:, -1, :] / temperature
            p = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(p, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return self.tok.decode(idx[0].tolist())


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
EOT = tokenizer.eos_token_id


def prepare_data(split, cache_path=None):
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
                ids.append(EOT)
            out_ids.append(ids)
            out_len.append(len(ids))
        return {"ids": out_ids, "len": out_len}

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names,
                       num_proc=8, desc="tokenizing")
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


def _slice(data, start, length):
    return torch.from_numpy(data[start: start + length].astype(np.int64))


def get_batch(data, block_size, batch_size, device, mode="standard", bag_size=1):
    """
    standard: x,y = [B, block_size]                      (next-token)
    full:     x   = [B, block_size, bag_size]  (bags to be embedding-averaged)
              y   = [B, block_size*bag_size]   (raw next-token labels; non-overlapping bags)
    output:   x   = [B, block_size]            (token-level input)
              y   = [B, block_size, bag_size]  (each position's next `bag_size` tokens)
    """
    if mode == "full":
        span = block_size * bag_size
        ix = torch.randint(len(data) - span - 1, (batch_size,))
        x = torch.stack([_slice(data, i, span) for i in ix]).view(batch_size, block_size, bag_size)
        y = torch.stack([_slice(data, i + 1, span) for i in ix])           # [B, span]
    elif mode == "output":
        span = block_size + bag_size
        ix = torch.randint(len(data) - span - 1, (batch_size,))
        x = torch.stack([_slice(data, i, block_size) for i in ix])         # [B, block_size]
        bags = []
        for i in ix:
            chunk = _slice(data, i, span)
            bags.append(torch.stack([chunk[1 + j: block_size + 1 + j] for j in range(bag_size)], dim=-1))
        y = torch.stack(bags)                                              # [B, block_size, bag_size]
    else:
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([_slice(data, i, block_size) for i in ix])
        y = torch.stack([_slice(data, i + 1, block_size) for i in ix])

    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def standard_loss(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def mce_loss_full(logits, y, bag_size):
    """Multi-hot CE for FULL superposition (Appendix A, Listing 3), fused.
    logits [B, T, V]; y [B, T*bag_size] raw next-token labels. Shift labels left
    by (bag_size-1), split into non-overlapping bags, average per-token CE over
    the bag (simplified MCE, Eq. 3). The vocab logsumexp is computed ONCE and
    reused across the bag instead of in `s` separate cross_entropy calls."""
    B, T, V = logits.shape
    offset = bag_size - 1
    pred = logits.reshape(B * T, V).float()                     # [N, V]
    lse = torch.logsumexp(pred, dim=-1, keepdim=True)           # [N, 1]  (once)
    y = F.pad(y, (0, offset), value=-100)[..., offset:].reshape(B * T, bag_size)
    mask = y != -100                                           # ignore padded tail
    chosen = pred.gather(1, y.clamp_min(0))                     # [N, bag]
    nll = (lse - chosen).masked_fill(~mask, 0.0)               # [N, bag]
    per_col = nll.sum(0) / mask.sum(0).clamp_min(1)            # per-slot mean over valid
    return per_col.mean()                                       # == (1/s) Σ_i CE_i


def mce_loss_output(logits, y_bags):
    """Multi-hot CE for OUTPUT-only superposition, fused: each position predicts
    its next `bag_size` tokens (overlapping), averaged. y_bags [B, T, bag_size].
    All targets are valid (no padding), so one mean over N*bag suffices."""
    B, T, V = logits.shape
    bag_size = y_bags.size(-1)
    pred = logits.reshape(B * T, V).float()                     # [N, V]
    lse = torch.logsumexp(pred, dim=-1, keepdim=True)           # [N, 1]  (once)
    chosen = pred.gather(1, y_bags.reshape(B * T, bag_size))    # [N, bag]
    return (lse - chosen).mean()                                # == (1/s) Σ_i CE_i


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters):
    """Always standard next-token loss, so val numbers stay comparable across phases."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device, mode="standard")
            with amp_autocast():
                logits = model(x)
                loss = standard_loss(logits, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------
def _rng_state():
    state = {"torch": torch.get_rng_state(),
             "numpy": np.random.get_state(),
             "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except Exception as e:
            print(f"  warning: CUDA RNG state not restored ({e})")


def save_checkpoint(model, optimizer, step, best_val, tag="latest"):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"ckpt_{tag}.pt")
    raw = getattr(model, "_orig_mod", model)   # unwrap torch.compile so keys are unprefixed
    payload = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "rng": _rng_state(),
        "config": {"BLOCK_SIZE": BLOCK_SIZE, "BATCH_SIZE": BATCH_SIZE,
                   "GRAD_ACCUM": GRAD_ACCUM, "MAX_STEPS": MAX_STEPS,
                   "TST_MODE": TST_MODE, "TST_BAG_SIZE": TST_BAG_SIZE,
                   "TST_RATIO": TST_RATIO, "DTYPE": DTYPE},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(model, optimizer, tag="latest"):
    path = os.path.join(CKPT_DIR, f"ckpt_{tag}.pt")
    if not os.path.exists(path):
        return 0, float("inf")
    print(f"resuming from {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    if cfg.get("BLOCK_SIZE") not in (None, BLOCK_SIZE):
        print(f"  warning: checkpoint BLOCK_SIZE={cfg['BLOCK_SIZE']} != current {BLOCK_SIZE}")
    raw = getattr(model, "_orig_mod", model)   # works whether or not model is compiled
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    try:
        _set_rng_state(ckpt["rng"])
    except Exception as e:
        print(f"  warning: could not restore RNG state ({e})")
    return ckpt["step"] + 1, ckpt.get("best_val", float("inf"))


def raw_tokens_seen(mode, bag, ratio, steps, batch, block):
    """Raw data tokens consumed over a run. full-superposition steps ingest
    `bag`x more tokens/step; output/off are token-level like the baseline."""
    if mode == "full":
        tst = int(ratio * steps)
        return tst * batch * block * bag + (steps - tst) * batch * block
    return steps * batch * block


def log_result(record, path):
    """Append one run as a JSON line (append-only; safe across parallel runs)."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"logged run '{record.get('run_name')}' -> {path}")


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(model, train_data, val_data):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    splits = {"train": train_data, "val": val_data}
    tst_steps = int(TST_RATIO * MAX_STEPS) if TST_MODE != "off" else 0

    start_step, best_val = (load_checkpoint(model, optimizer) if RESUME else (0, float("inf")))
    if start_step >= MAX_STEPS:
        print(f"checkpoint already at step {start_step} >= MAX_STEPS; nothing to do.")
        return None
    if TST_MODE != "off":
        print(f"TST: mode={TST_MODE} s={TST_BAG_SIZE} r={TST_RATIO} "
              f"-> superposition for steps [0, {tst_steps}), recovery after.")

    best_step, last_val = -1, float("nan")
    t0 = time.time()
    model.train()
    pbar = tqdm(range(start_step, MAX_STEPS), initial=start_step, total=MAX_STEPS, desc="training")
    step = start_step
    prev_in_tst = start_step < tst_steps
    try:
        for step in pbar:
            in_tst = step < tst_steps
            if prev_in_tst and not in_tst:
                print(f"\n--- step {step}: switching to recovery (standard next-token) ---")
            prev_in_tst = in_tst
            mode = TST_MODE if in_tst else "standard"
            bag = TST_BAG_SIZE if in_tst else 1

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            for _ in range(GRAD_ACCUM):
                x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE, mode=mode, bag_size=bag)
                with amp_autocast():
                    logits = model(x)
                    if mode == "full":
                        loss = mce_loss_full(logits, y, bag)
                    elif mode == "output":
                        loss = mce_loss_output(logits, y)
                    else:
                        loss = standard_loss(logits, y)
                (loss / GRAD_ACCUM).backward()   # backward outside autocast

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.2e}",
                             phase=("tst" if in_tst else "rec"))

            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
                stats = estimate_loss(model, splits, BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS)
                last_val = stats["val"]
                tag = " (superposition; val NTP not yet meaningful)" if in_tst else ""
                print(f"\nstep {step}: train {stats['train']:.4f} | val {stats['val']:.4f} "
                      f"| val ppl {math.exp(stats['val']):.2f}{tag}")
                save_checkpoint(model, optimizer, step, best_val, tag="latest")
                # only track "best" during recovery, where val NTP is meaningful
                if not in_tst and stats["val"] < best_val:
                    best_val, best_step = stats["val"], step
                    save_checkpoint(model, optimizer, step, best_val, tag="best")
                    print(f"  new best val {best_val:.4f} -> ckpt_best.pt")
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {step}; saving before exit...")
        save_checkpoint(model, optimizer, step, best_val, tag="latest")
        print("saved checkpoints/ckpt_latest.pt — rerun to resume.")
        raise

    save_checkpoint(model, optimizer, MAX_STEPS - 1, best_val, tag="latest")
    print(f"done. best val loss {best_val:.4f} (ppl {math.exp(best_val):.2f})")
    return {
        "best_val": best_val,
        "best_step": best_step,
        "final_val": last_val,
        "final_step": step,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a small GPT-2 on WikiText-103, optionally with TST.")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS,
                   help="total optimizer steps; raise to continue a finished run")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--min-lr", type=float, default=MIN_LR)
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--ckpt-dir", type=str, default=CKPT_DIR)
    p.add_argument("--no-resume", action="store_true", help="start fresh, ignoring any checkpoint")
    p.add_argument("--tst-mode", choices=["off", "full", "output"], default=TST_MODE,
                   help="off = baseline; full = input+output superposition; output = output-only")
    p.add_argument("--tst-bag-size", type=int, default=TST_BAG_SIZE, help="superposition bag size s")
    p.add_argument("--tst-ratio", type=float, default=TST_RATIO,
                   help="fraction of steps in the superposition phase r")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default=DTYPE,
                   help="bf16 enables autocast (faster, MPS/CUDA); fp32 for reproducible baselines")
    p.add_argument("--compile", action="store_true", help="wrap model in torch.compile (CUDA)")
    p.add_argument("--run-name", type=str, default=RUN_NAME,
                   help="label for the log record (defaults to the checkpoint dir name)")
    p.add_argument("--log-file", type=str, default=LOG_FILE, help="JSONL results log to append to")
    args = p.parse_args()

    MAX_STEPS = args.max_steps
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM = args.grad_accum
    BLOCK_SIZE = args.block_size
    LR = args.lr
    MIN_LR = args.min_lr
    WARMUP_STEPS = args.warmup_steps
    EVAL_INTERVAL = args.eval_interval
    CKPT_INTERVAL = EVAL_INTERVAL
    CKPT_DIR = args.ckpt_dir
    RESUME = not args.no_resume
    TST_MODE = args.tst_mode
    TST_BAG_SIZE = args.tst_bag_size
    TST_RATIO = args.tst_ratio
    DTYPE = args.dtype
    USE_AMP = (DTYPE == "bf16")
    COMPILE = args.compile
    LOG_FILE = args.log_file
    RUN_NAME = args.run_name or os.path.basename(CKPT_DIR.rstrip("/")) or "run"

    train_data = prepare_data("train")
    val_data = prepare_data("validation")

    model = GPT2(max_len=BLOCK_SIZE).to(DEVICE)
    raw_model = model                          # keep an uncompiled handle for generate/state_dict
    if COMPILE:
        if DEVICE != "cuda":
            print(f"warning: --compile on device '{DEVICE}'; torch.compile is most reliable on CUDA.")
        print("compiling with torch.compile (first step + first eval warm up slowly)...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"model parameters: {n_params/1e6:.1f}M | device: {DEVICE} | dtype {DTYPE} | "
          f"compile {COMPILE} | max_steps {MAX_STEPS} | batch {BATCH_SIZE}x{GRAD_ACCUM} | "
          f"block {BLOCK_SIZE} | tst {TST_MODE}")

    results = train(model, train_data, val_data)

    if results is not None:
        tokens = raw_tokens_seen(TST_MODE, TST_BAG_SIZE, TST_RATIO, MAX_STEPS, BATCH_SIZE, BLOCK_SIZE)
        bv = results["best_val"]
        record = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_name": RUN_NAME, "ckpt_dir": CKPT_DIR,
            "device": DEVICE, "dtype": DTYPE, "compile": COMPILE,
            "n_params_m": round(n_params / 1e6, 1),
            "max_steps": MAX_STEPS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "block_size": BLOCK_SIZE, "lr": LR, "min_lr": MIN_LR, "warmup_steps": WARMUP_STEPS,
            "tst_mode": TST_MODE, "tst_bag_size": TST_BAG_SIZE, "tst_ratio": TST_RATIO,
            "raw_tokens_seen": tokens,
            "epochs": round(tokens / len(train_data), 3),
            "best_val": round(bv, 4),
            "best_val_ppl": round(math.exp(bv), 2) if bv < float("inf") else None,
            "best_step": results["best_step"],
            "final_val": round(results["final_val"], 4),
            "elapsed_min": results["elapsed_min"],
        }
        log_result(record, LOG_FILE)

    print("Generated:", raw_model.generate("The future of AI is", max_new=40))
