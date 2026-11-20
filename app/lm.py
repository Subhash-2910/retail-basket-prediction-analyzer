"""Milestone 3: a small language model pretrained on basket sequences, in PyTorch on the CPU.

Follows the pretraining recipe from 'Teach a Language Model What Sells Together': the plain next-token
objective, every basket shown in a fresh random order each epoch, AdamW, linear warmup then a cosine decay to
a tenth of the peak rate, gradient clipping, and a held-out validation loss. (The course library trains with
MLX on a Mac GPU; this is the same recipe in PyTorch so it runs anywhere, including Docker and CI.)

Each item is one token. There are deliberately no position embeddings: item order inside a basket is random,
so the model should treat the items seen so far as a set. The causal mask still lets it read only the past.
"""
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PAD, BOS = 0, 1  # token ids; items start at 2


class BasketLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, dim=dim, layers=layers, heads=heads, dropout=dropout)
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied input/output embeddings: fewer parameters
        nn.init.normal_(self.embed.weight, std=0.02)  # small start, as in GPT-style models

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = ids.shape[1]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)  # True = may not look
        x = self.blocks(self.embed(ids), mask=causal, src_key_padding_mask=ids == PAD, is_causal=True)
        return self.head(self.norm(x))


class LMPredictor:
    """Wraps a trained BasketLM so it answers the same questions as NGramModel (scores, predict, ranks)."""

    def __init__(self, model: BasketLM, vocab: list[str], max_len: int = 32):
        self.model, self.vocab, self.max_len = model.eval(), vocab, max_len
        self.index = {item: i for i, item in enumerate(vocab)}

    def _ids(self, seq):
        return [BOS] + [self.index[i] + 2 if i in self.index else PAD for i in seq[: self.max_len]]

    @torch.no_grad()
    def scores(self, prefix: list[str]) -> np.ndarray:
        logits = self.model(torch.tensor([self._ids(prefix)]))[0, -1, 2:].numpy().astype(float)
        for item in prefix:
            if item in self.index:
                logits[self.index[item]] = -np.inf
        return logits

    def predict(self, prefix: list[str], k: int = 5) -> list[str]:
        return [self.vocab[i] for i in np.argsort(-self.scores(prefix), kind="stable")[:k]]

    @torch.no_grad()
    def ranks(self, sequences: list[list[str]], batch_size: int = 256) -> list[np.ndarray]:
        """Rank of the true item at positions 1..len-1 of each basket; one forward pass per batch (causal)."""
        out = []
        for start in range(0, len(sequences), batch_size):
            batch = [s[: self.max_len] for s in sequences[start:start + batch_size]]
            ids = _pad([self._ids(s) for s in batch])
            logits = self.model(ids)[:, :, 2:]  # (B, T, items): position t predicts item t
            items = (ids[:, 1:] - 2).clamp(min=0)  # (B, T-1) item index at each position
            real = ids[:, 1:] >= 2
            seen = torch.zeros(logits.shape[0], logits.shape[1], logits.shape[2], dtype=torch.bool)
            onehot = torch.zeros_like(seen[:, 1:])
            onehot.scatter_(2, items.unsqueeze(-1), real.unsqueeze(-1))
            seen[:, 1:] = onehot.cumsum(1) > 0  # items already in the basket before each position
            logits = logits.masked_fill(seen, float("-inf"))
            for b, seq in enumerate(batch):
                r = []
                for t in range(1, len(seq)):
                    row = logits[b, t]
                    if seq[t] in self.index:
                        r.append(int((row > row[self.index[seq[t]]]).sum()) + 1)
                    else:
                        r.append(int(torch.isfinite(row).sum()) + 1)
                out.append(np.array(r, dtype=int))
        return out


def _pad(rows: list[list[int]]) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([r + [PAD] * (width - len(r)) for r in rows], dtype=torch.long)


def _loss(model, ids):
    logits = model(ids[:, :-1])
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                       ignore_index=PAD)


@torch.no_grad()
def validation_loss(model, rows: list[list[int]], batch_size: int = 256) -> float:
    model.eval()
    total = count = 0.0
    for start in range(0, len(rows), batch_size):
        ids = _pad(rows[start:start + batch_size])
        n = float((ids[:, 1:] != PAD).sum())
        total += float(_loss(model, ids)) * n
        count += n
    model.train()
    return total / count


def pretrain(train: list[list[str]], val: list[list[str]] | None = None, epochs: int = 10, batch_size: int = 128,
             learning_rate: float = 3e-3, warmup: int = 100, max_len: int = 32, seed: int = 0, log=print,
             **model_args) -> tuple[LMPredictor, list[dict]]:
    """Train a BasketLM from scratch on `train`; returns the predictor and the loss history."""
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = random.Random(seed)
    vocab = sorted({item for b in train for item in b})
    index = {item: i + 2 for i, item in enumerate(vocab)}
    model = BasketLM(len(vocab) + 2, **model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / batch_size)

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    val_rows = None
    if val:
        vr = random.Random(seed + 1)
        val_rows = [[BOS] + [index[i] for i in vr.sample(b, len(b)) if i in index][:max_len] for b in val]

    history, started, step = [], time.time(), 0
    if val_rows:
        history.append({"epoch": 0, "step": 0, "loss": None, "val_loss": validation_loss(model, val_rows)})
        log(f"epoch 0  val loss {history[-1]['val_loss']:.3f}")
    model.train()
    for epoch in range(1, epochs + 1):
        rows = [[BOS] + [index[i] for i in rng.sample(b, len(b))][:max_len] for b in train]  # fresh order
        rows.sort(key=len)  # batches of similar length waste little padding
        batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        rng.shuffle(batches)
        losses = []
        for batch in batches:
            loss = _loss(model, _pad(batch))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(loss.item())
        val_loss = validation_loss(model, val_rows) if val_rows else None
        history.append({"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "val_loss": val_loss,
                        "seconds": round(time.time() - started, 1)})
        log(f"epoch {epoch}  train loss {history[-1]['loss']:.3f}"
            + (f"  val loss {val_loss:.3f}" if val_loss is not None else "")
            + f"  ({history[-1]['seconds']}s)")
    return LMPredictor(model, vocab, max_len), history


def save(predictor: LMPredictor, path="models/lm.pt") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": predictor.model.config, "state": predictor.model.state_dict(),
                "vocab": predictor.vocab, "max_len": predictor.max_len}, path)
    return path


def load(path="models/lm.pt") -> LMPredictor:
    if not Path(path).exists():
        raise FileNotFoundError(f"no trained model at {path} - run `python -m app.cli train` first")
    saved = torch.load(path, weights_only=False)
    model = BasketLM(**saved["config"])
    model.load_state_dict(saved["state"])
    return LMPredictor(model, saved["vocab"], saved["max_len"])


def n_parameters(predictor: LMPredictor) -> int:
    return sum(p.numel() for p in predictor.model.parameters())
