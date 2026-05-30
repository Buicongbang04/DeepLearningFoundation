# Project 06 — Autoencoder + Anomaly Detection

A small convolutional autoencoder trained on a single Fashion-MNIST
class (`T-shirt/top`) and then used as an **anomaly scorer** on the
rest of the validation set. The sixth end-to-end project of the
course, paired with Chapter 14.

**Companion chapter:** `docs/13_autoencoders_representation_learning.md`.

## What this project demonstrates

- A small **conv autoencoder** (`ConvAutoencoder` in `src/models.py`,
  ~112k parameters) that maps 28×28 → 7×7 feature map → 32-D latent →
  reconstruction.
- The "single-class training + cross-class scoring" recipe used in
  production anomaly-detection systems.
- **F1 / precision / recall** on a known-anomaly subset, with a tuned
  threshold and a per-class MSE histogram saved as a figure.

The driver lives in this folder (`train_and_score.py`) rather than the
shared `src/train.py` because the loss is per-image MSE rather than
classification cross-entropy, and the post-training step is a *score*
computation, not a metric.

## Reproduce

From the repo root:

```bash
pip install -r requirements.txt

python projects/project_06_autoencoder_anomaly_detection/train_and_score.py \
    --config configs/autoencoder_fashion_mnist.yaml
```

Fashion-MNIST (~30 MB) downloads to `datasets/fashion_mnist/` on
first run.

## Configuration highlights

`configs/autoencoder_fashion_mnist.yaml`:

- `data.normal_class: 0` — `T-shirt/top` only for training.
- `data.anomaly_classes: [1, 5, 9]` — `Trouser`, `Sandal`, `Ankle boot`
  used for scoring. They look obviously different from T-shirts, so a
  small autoencoder should reconstruct them badly.
- `model.name: conv_autoencoder`, `latent_dim: 32`.
- `optim: adam`, `lr: 1e-3`. 5 epochs.

## Outputs

After `train_and_score.py`:

```
experiments/autoencoder_fashion_mnist/
├── checkpoints/
│   ├── best.pt              # best-by-val-reconstruction-MSE
│   └── last.pt
└── logs/
    └── history.json         # train/val MSE per epoch + anomaly metrics
projects/project_06_autoencoder_anomaly_detection/
└── figures/
    └── score_histogram.png  # per-image MSE histogram, normal vs anomaly
```

`experiments/` is gitignored; `figures/` is gitignored per the
repo-wide `projects/*/figures/*.png` rule.

## Files

- `configs/autoencoder_fashion_mnist.yaml` — single source of truth.
- `src/datasets.py` — `fashion_mnist` loader registered alongside MNIST.
- `src/models.py` — `ConvAutoencoder` definition and registry entry.
- `projects/project_06_autoencoder_anomaly_detection/train_and_score.py`
  — local driver (train + score + figure).
- `projects/project_06_autoencoder_anomaly_detection/report.md` —
  written report with the actual numbers.
