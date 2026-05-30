# Project 02 — SmallCNN on CIFAR-10

A small `Conv → BN → ReLU → MaxPool` CNN trained on CIFAR-10. The second
end-to-end project of the course, paired with Chapter 8 (CNN basics).

**Companion chapter:** `docs/07_cnn_basics.md`.

## What this project demonstrates

- A 3-block CNN (`SmallCNN` in `src/models.py`) at ~95k parameters.
- The full augmentation → `Conv → BN → ReLU → MaxPool` → `AdaptiveAvgPool`
  → `Linear` pipeline from the chapter doc.
- That a small CNN with the right inductive bias *and* basic augmentation
  beats an MLP at the same parameter count by a wide margin on CIFAR-10.

## Reproduce

From the repo root:

```bash
pip install -r requirements.txt

python src/train.py     --config configs/cnn_cifar10.yaml
python src/evaluate.py  --config configs/cnn_cifar10.yaml \
                        --checkpoint experiments/cnn_cifar10/checkpoints/best.pt
```

CIFAR-10 (~170 MB) will download to `datasets/cifar10/` on first run.

## Configuration highlights

`configs/cnn_cifar10.yaml`:

- `model.name: small_cnn`, channels `(32, 64, 128)`, dropout `0.2`.
- `optim: adamw`, `lr: 1e-3`, `weight_decay: 5e-4`.
- `data.augment: true` — random crop with padding 4 + horizontal flip on
  the training set only.
- `training.num_epochs: 10`.

## Outputs

After `train.py`:

```
experiments/cnn_cifar10/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── logs/
    └── history.json
```

Both directories are gitignored.

## Files

- `configs/cnn_cifar10.yaml` — single source of truth.
- `src/datasets.py` — `_cifar10_loaders` factory with augmentation knob.
- `src/models.py` — `SmallCNN` definition and registry entry.
- `src/train.py` / `src/evaluate.py` / `src/inference.py` — shared CLI
  scripts (unchanged from Project 01).
- `projects/project_02_cnn_cifar10/report.md` — written report with
  actual training numbers.
