"""Train a model from a YAML config.

Usage:
    python src/train.py --config configs/mlp_mnist.yaml

The script saves the best checkpoint (by validation accuracy) to
``<output.checkpoint_dir>/best.pt`` and the last checkpoint to ``last.pt``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Allow `python src/train.py` from the repo root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import build_loaders
from models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["name"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(),
                                lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(),
                                 lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(),
                               lr=cfg["lr"],
                               momentum=cfg.get("momentum", 0.9),
                               weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer: {name!r}")


def train_one_epoch(model, loader, criterion, optimizer, device, log_every) -> float:
    model.train()
    running_loss, n_seen = 0.0, 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        if (batch_idx + 1) % log_every == 0:
            print(f"    step {batch_idx + 1:5d}: loss={running_loss / n_seen:.4f}")
    return running_loss / n_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"device: {device}")

    train_loader, val_loader, info = build_loaders(cfg)
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg['model']['name']}  params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg["optim"])

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, -1
    log_every = cfg["training"]["log_every"]

    for epoch in range(cfg["training"]["num_epochs"]):
        t0 = time.time()
        print(f"\n=== epoch {epoch + 1} / {cfg['training']['num_epochs']} ===")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, log_every)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({dt:.1f}s)")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "config": cfg,
                "classes": info["classes"],
            }, ckpt_dir / "best.pt")
            print(f"  -> saved best @ epoch {epoch + 1}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "config": cfg,
            "classes": info["classes"],
        }, ckpt_dir / "last.pt")

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
        "config": cfg,
    }
    (log_dir / "history.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest val_acc: {best_val_acc:.4f} @ epoch {best_epoch + 1}")
    print(f"saved: {ckpt_dir / 'best.pt'}")
    print(f"saved: {log_dir / 'history.json'}")


if __name__ == "__main__":
    main()
