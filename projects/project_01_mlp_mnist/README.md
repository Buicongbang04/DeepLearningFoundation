# Project 01 — MLP on MNIST

The first end-to-end project of the course. A simple MLP, trained on MNIST,
with the standard PyTorch training loop, best-checkpoint saving by validation
accuracy, evaluation, and CLI inference.

**Companion chapter:** `docs/03_training_loop.md` (Chapter 4).

## What this project demonstrates

- A clean `Dataset` / `DataLoader` setup via `src/datasets.py`.
- An `nn.Module` defined in `src/models.py` (`MLPClassifier`).
- The canonical training loop in `src/train.py` with `model.train()` /
  `model.eval()`, validation, gradient zeroing, and best-checkpoint saving.
- A reload + re-evaluate path in `src/evaluate.py` that confirms the saved
  metric.
- A CLI inference path in `src/inference.py` that takes a single image (or a
  folder) and writes JSONL predictions.

## Reproduce the headline result

From the repo root:

```bash
pip install -r requirements.txt

python src/train.py     --config configs/mlp_mnist.yaml
python src/evaluate.py  --config configs/mlp_mnist.yaml \
                        --checkpoint experiments/mlp_mnist/checkpoints/best.pt
python src/inference.py --checkpoint experiments/mlp_mnist/checkpoints/best.pt \
                        --input <a-28x28-png-of-a-digit>
```

The training run takes a few minutes on CPU and well under a minute on a
modest GPU. MNIST will be downloaded automatically to `datasets/mnist/`
on the first run.

## Configuration

Everything that determines a run is in `configs/mlp_mnist.yaml`. Notable
fields:

- `seed: 42` — reproducibility.
- `model.hidden_dim: 128` — single hidden layer.
- `optim.lr: 0.001` with `adam`.
- `training.num_epochs: 3` — enough to reach ≥ 97% validation accuracy.

To run an ablation (e.g., larger hidden size, different optimizer, more
epochs), copy `configs/mlp_mnist.yaml` to a new file and point `train.py`
at it.

## Outputs

After `train.py` finishes:

```
experiments/mlp_mnist/
├── checkpoints/
│   ├── best.pt              # checkpoint of the epoch with best val accuracy
│   └── last.pt              # checkpoint at the final epoch
└── logs/
    └── history.json         # per-epoch train/val loss and accuracy
```

The contents of `experiments/` are gitignored — every run regenerates them.

## Files

- `configs/mlp_mnist.yaml` — single source of truth for the run.
- `src/datasets.py` — `build_loaders(config)` factory for MNIST loaders.
- `src/models.py` — `MLPClassifier` and `build_model(config)` factory.
- `src/train.py` — argparse CLI, training loop, best-checkpoint save.
- `src/evaluate.py` — load checkpoint, re-evaluate on val set.
- `src/inference.py` — load checkpoint, predict on PNG/JPG input, write JSONL.
- `projects/project_01_mlp_mnist/report.md` — the written report.
