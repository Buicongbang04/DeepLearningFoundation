"""Evaluate a saved checkpoint on the validation set of a config.

Usage:
    python src/evaluate.py --config configs/mlp_mnist.yaml \\
        --checkpoint experiments/mlp_mnist/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import build_loaders
from models import build_model
from train import evaluate, resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"device: {device}")

    _, val_loader, info = build_loaders(cfg)
    print(f"val batches: {len(val_loader)}")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded checkpoint: epoch={ckpt.get('epoch', '?')}  "
          f"reported val_acc={ckpt.get('val_acc', '?')}")

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"\nre-evaluated val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
