"""End-to-end driver for Project 06.

Single script because the loss surface (MSE reconstruction over images,
not classification) and the post-training "anomaly scoring" step differ
enough from the shared `src/train.py` that wrapping it in the generic
training driver would obscure the chapter's point.

What this script does, in order:

1. Load Fashion-MNIST.
2. Subset the *training* set to a single "normal" class.
3. Train a conv autoencoder for `training.num_epochs` epochs.
4. Score the *whole* validation set by per-image reconstruction MSE.
5. Compute precision/recall + a histogram (saved as PNG) for the
   "normal vs anomaly" detection task using the held-out anomaly
   classes from the config.
6. Save best.pt + history.json + scores.json + figures/ outputs into
   the canonical experiments/ layout.

Run from the repo root:

    python projects/project_06_autoencoder_anomaly_detection/train_and_score.py \\
        --config configs/autoencoder_fashion_mnist.yaml
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _by_class_indices(ds, target_class: int) -> list[int]:
    return [i for i, t in enumerate(ds.targets) if int(t) == target_class]


def _by_classes_indices(ds, target_classes: list[int]) -> list[int]:
    targets_set = set(int(c) for c in target_classes)
    return [i for i, t in enumerate(ds.targets) if int(t) in targets_set]


@torch.no_grad()
def reconstruction_mse_per_image(model, loader, device):
    model.eval()
    mses = []
    for x, _ in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        per = (x_hat - x).pow(2).mean(dim=[1, 2, 3])
        mses.append(per.cpu())
    return torch.cat(mses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"device: {device}")

    data_cfg   = cfg["data"]
    train_cfg  = cfg["training"]
    out_cfg    = cfg["output"]

    normal_class = int(data_cfg["normal_class"])
    anomaly_classes = list(int(c) for c in data_cfg["anomaly_classes"])

    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.FashionMNIST(data_cfg["root"], train=True,  download=True, transform=transform)
    full_val   = datasets.FashionMNIST(data_cfg["root"], train=False, download=True, transform=transform)
    print(f"classes: {full_train.classes}")
    print(f"normal class : {normal_class}  ({full_train.classes[normal_class]})")
    print(f"anomaly classes: {anomaly_classes}  "
          f"({[full_train.classes[i] for i in anomaly_classes]})")

    normal_train_idx = _by_class_indices(full_train, normal_class)
    train_loader = DataLoader(Subset(full_train, normal_train_idx),
                              batch_size=data_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"])
    print(f"train (normal-class only) size: {len(normal_train_idx)}")

    normal_val_idx  = _by_class_indices(full_val, normal_class)
    anomaly_val_idx = _by_classes_indices(full_val, anomaly_classes)
    val_loader_normal  = DataLoader(Subset(full_val, normal_val_idx),
                                    batch_size=data_cfg["batch_size"], shuffle=False,
                                    num_workers=data_cfg["num_workers"])
    val_loader_anomaly = DataLoader(Subset(full_val, anomaly_val_idx),
                                    batch_size=data_cfg["batch_size"], shuffle=False,
                                    num_workers=data_cfg["num_workers"])
    print(f"val normal: {len(normal_val_idx)}, val anomaly: {len(anomaly_val_idx)}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg['model']['name']}  params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg["optim"]["lr"],
                           weight_decay=cfg["optim"].get("weight_decay", 0.0))

    ckpt_dir = ROOT / out_cfg["checkpoint_dir"]
    log_dir  = ROOT / out_cfg["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir  = ROOT / "projects" / "project_06_autoencoder_anomaly_detection" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss_normal": []}
    best_val_loss = float("inf")

    log_every = train_cfg["log_every"]
    for epoch in range(train_cfg["num_epochs"]):
        t0 = time.time()
        print(f"\n=== epoch {epoch + 1}/{train_cfg['num_epochs']} ===")
        model.train()
        running, n = 0.0, 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            opt.zero_grad()
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward(); opt.step()
            running += loss.item() * x.size(0); n += x.size(0)
            if (batch_idx + 1) % log_every == 0:
                print(f"    step {batch_idx + 1:5d}: loss={running / n:.5f}")
        train_loss = running / n

        val_mses = reconstruction_mse_per_image(model, val_loader_normal, device)
        val_loss = float(val_mses.mean().item())

        history["train_loss"].append(train_loss)
        history["val_loss_normal"].append(val_loss)
        print(f"  train_loss={train_loss:.5f}  val_loss(normal)={val_loss:.5f}  ({time.time() - t0:.1f}s)")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg,
            }, ckpt_dir / "best.pt")
            print(f"  -> saved best @ epoch {epoch + 1}")
        torch.save({"model_state_dict": model.state_dict(), "config": cfg},
                   ckpt_dir / "last.pt")

    # --- Anomaly scoring on the held-out val set ---
    print("\n=== scoring anomaly vs normal on val ===")
    scores_normal  = reconstruction_mse_per_image(model, val_loader_normal,  device)
    scores_anomaly = reconstruction_mse_per_image(model, val_loader_anomaly, device)
    all_scores = torch.cat([scores_normal, scores_anomaly])
    labels     = torch.cat([torch.zeros_like(scores_normal),
                            torch.ones_like(scores_anomaly)])

    # Pick the threshold that maximizes F1 on a coarse grid.
    thresholds = torch.linspace(all_scores.min(), all_scores.max(), 200)
    best_f1, best_t = 0.0, float(thresholds[0])
    best_prec, best_rec = 0.0, 0.0
    for t in thresholds:
        pred = (all_scores > t).float()
        tp = ((pred == 1) & (labels == 1)).sum().item()
        fp = ((pred == 1) & (labels == 0)).sum().item()
        fn = ((pred == 0) & (labels == 1)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
            best_prec, best_rec = prec, rec
    print(f"  best F1   = {best_f1:.4f}  (threshold MSE = {best_t:.5f})")
    print(f"  precision = {best_prec:.4f}, recall = {best_rec:.4f}")
    print(f"  mean MSE  normal={scores_normal.mean():.5f}  anomaly={scores_anomaly.mean():.5f}")

    # --- Save outputs ---
    summary = {
        "best_val_loss_normal": best_val_loss,
        "history": history,
        "anomaly_detection": {
            "best_f1": best_f1,
            "best_threshold_mse": best_t,
            "best_precision": best_prec,
            "best_recall": best_rec,
            "mean_mse_normal":  float(scores_normal.mean()),
            "mean_mse_anomaly": float(scores_anomaly.mean()),
            "num_val_normal":   int(scores_normal.numel()),
            "num_val_anomaly":  int(scores_anomaly.numel()),
        },
        "config": cfg,
    }
    (log_dir / "history.json").write_text(json.dumps(summary, indent=2))

    # Histogram figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(scores_normal.numpy(),  bins=40, alpha=0.6,
                label=f"normal (mean={scores_normal.mean():.4f})")
        ax.hist(scores_anomaly.numpy(), bins=40, alpha=0.6,
                label=f"anomaly (mean={scores_anomaly.mean():.4f})")
        ax.axvline(best_t, color="red", ls="--", label=f"best threshold = {best_t:.4f}")
        ax.set_xlabel("reconstruction MSE per image"); ax.set_ylabel("count")
        ax.set_title("Anomaly score distribution on Fashion-MNIST val")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = fig_dir / "score_histogram.png"
        plt.savefig(fig_path, dpi=120); plt.close(fig)
        print(f"  saved figure: {fig_path}")
    except Exception as e:
        print(f"  (skipped figure: {e})")

    print(f"\nbest val_loss_normal: {best_val_loss:.5f}")
    print(f"saved: {ckpt_dir / 'best.pt'}")
    print(f"saved: {log_dir / 'history.json'}")


if __name__ == "__main__":
    main()
