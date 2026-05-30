"""Training driver for text-classification (LSTM) and char-level LM
(decoder-only Transformer). Same config-driven pattern as ``train.py``,
adapted to the text dataloaders in ``datasets.py`` /
``_text_data.py``.

The dataloader interface is what differs:
- ``synthetic_text`` yields ``(ids, lengths, label)`` triples.
- ``char_corpus``    yields ``(input_ids, target_ids)`` pairs where
  ``target_ids = input_ids`` shifted by one position.

This module dispatches on ``data.name`` to pick the right per-batch
forward + loss path, then reuses the same checkpoint / history layout.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import build_loaders
from models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["name"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                               momentum=cfg.get("momentum", 0.9),
                               weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer: {name!r}")


def step_text_classifier(model, batch, device):
    x, lengths, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x, lengths=lengths)
    return logits, y


def step_lm(model, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)                                       # (B, T, V)
    return logits, y


def loss_text_classifier(logits, y, criterion):
    return criterion(logits, y)


def loss_lm(logits, y, criterion):
    B, T, V = logits.shape
    return criterion(logits.reshape(B * T, V), y.reshape(B * T))


def metric_text_classifier(logits, y):
    return (logits.argmax(dim=-1) == y).sum().item(), y.size(0)


def metric_lm(logits, y):
    # Per-token accuracy (proxy for LM quality alongside loss).
    pred = logits.argmax(dim=-1)
    return (pred == y).sum().item(), y.numel()


def train_one_epoch(model, loader, criterion, optimizer, device,
                    log_every, step_fn, loss_fn, clip_grad=None):
    model.train()
    running_loss, n_seen = 0.0, 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        logits, y = step_fn(model, batch, device)
        loss = loss_fn(logits, y, criterion)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        size = y.size(0)
        running_loss += loss.item() * size
        n_seen += size
        if (batch_idx + 1) % log_every == 0:
            print(f"    step {batch_idx + 1:5d}: loss={running_loss / n_seen:.4f}")
    return running_loss / n_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, step_fn, loss_fn, metric_fn):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    n_loss = 0
    for batch in loader:
        logits, y = step_fn(model, batch, device)
        loss = loss_fn(logits, y, criterion)
        size = y.size(0)
        loss_sum += loss.item() * size
        n_loss += size
        c, t = metric_fn(logits, y)
        correct += c; total += t
    return loss_sum / max(n_loss, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"device: {device}")

    train_loader, val_loader, info = build_loaders(cfg)

    # Auto-fill dataset-derived knobs into model config.
    mcfg = cfg["model"]
    if cfg["data"]["name"] == "synthetic_text":
        mcfg.setdefault("vocab_size", info["vocab_size"])
        mcfg.setdefault("num_classes", info["num_classes"])
        mcfg.setdefault("pad_id", info["pad_id"])
        step_fn, loss_fn, metric_fn = step_text_classifier, loss_text_classifier, metric_text_classifier
    elif cfg["data"]["name"] == "char_corpus":
        mcfg.setdefault("vocab_size", info["vocab_size"])
        mcfg.setdefault("block_size", info["block_size"])
        step_fn, loss_fn, metric_fn = step_lm, loss_lm, metric_lm
    else:
        raise ValueError(f"train_text.py: unsupported data.name "
                         f"{cfg['data']['name']!r}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {mcfg['name']}  params: {n_params:,}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg["optim"])

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    extra = {}
    for k in ("classes", "stoi", "itos", "max_len", "pad_id", "block_size", "vocab_size"):
        if k in info:
            extra[k] = info[k]

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1
    log_every = cfg["training"]["log_every"]
    clip_grad = cfg["training"].get("clip_grad", None)

    for epoch in range(cfg["training"]["num_epochs"]):
        t0 = time.time()
        print(f"\n=== epoch {epoch + 1} / {cfg['training']['num_epochs']} ===")
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, log_every,
                                     step_fn, loss_fn, clip_grad)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device,
                                     step_fn, loss_fn, metric_fn)
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  ({dt:.1f}s)")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": cfg,
                **extra,
            }, ckpt_dir / "best.pt")
            print(f"  -> saved best @ epoch {epoch + 1}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": cfg,
            **extra,
        }, ckpt_dir / "last.pt")

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "config": cfg,
    }
    (log_dir / "history.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest val_loss: {best_val_loss:.4f} @ epoch {best_epoch + 1}")
    print(f"saved: {ckpt_dir / 'best.pt'}")
    print(f"saved: {log_dir / 'history.json'}")


if __name__ == "__main__":
    main()
