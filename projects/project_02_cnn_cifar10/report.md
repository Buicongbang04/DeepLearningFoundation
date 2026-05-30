# Project 02 — SmallCNN on CIFAR-10: Report

## Problem

CIFAR-10 image classification — 10 classes, 50k train / 10k test, RGB
32 × 32. Metric: **top-1 validation accuracy** on the official test set,
with cross-entropy loss as the training objective.

The point of this project is to apply the CNN inductive bias (local
connectivity + parameter sharing + small kernels stacked deep) on a
genuinely *natural-image* dataset, after seeing it succeed on MNIST in
Chapter 8's companion notebook.

## Method

`SmallCNN` from `src/models.py`: three `Conv → BN → ReLU → MaxPool`
blocks with channel ramp `(32, 64, 128)`, followed by
`AdaptiveAvgPool2d((1, 1))` → `Dropout(0.2)` → `Linear(128, 10)`. Total
**94,762 parameters**.

Training recipe (single source of truth: `configs/cnn_cifar10.yaml`):

- **Optimizer:** AdamW, `lr = 1e-3`, `weight_decay = 5e-4`.
- **Augmentation (train only):** `RandomCrop(32, padding=4)` +
  `RandomHorizontalFlip()`.
- **Normalization:** CIFAR-10 channel mean `(0.4914, 0.4822, 0.4465)`,
  std `(0.2470, 0.2435, 0.2616)`.
- **Batch size:** 128, 10 epochs, `seed: 42`.

## Results

### Headline metric

| Split        | Loss   | Accuracy |
|--------------|--------|----------|
| Train (ep 10)| 0.9227 | —        |
| **Val (ep 10, best)** | **0.9988** | **65.15%** |

### Training curve (per-epoch)

| Epoch | Train loss | Val loss | Val acc | Best? |
|------:|-----------:|---------:|--------:|:----:|
| 1     | 1.5514     | 1.4561   | 47.56%  | ✓    |
| 2     | 1.2709     | 1.1881   | 57.28%  | ✓    |
| 3     | 1.1857     | 1.4345   | 50.06%  |      |
| 4     | 1.1161     | 1.0601   | 62.29%  | ✓    |
| 5     | 1.0731     | 1.0207   | 63.72%  | ✓    |
| 6     | 1.0324     | 1.0016   | 63.95%  | ✓    |
| 7     | 0.9983     | 0.9897   | 64.95%  | ✓    |
| 8     | 0.9707     | 1.0006   | 64.76%  |      |
| 9     | 0.9462     | 1.0977   | 61.66%  |      |
| 10    | 0.9227     | 0.9988   | **65.15%** | ✓ |

### Wall-clock

~20 seconds per epoch on CPU. 10 epochs end-to-end take ~3.5 minutes
including CIFAR-10 download (~170 MB on the first run).

### Interpretation

The training loss decreases monotonically from 1.55 to 0.92, and the
validation loss tracks reasonably well — the train-val gap is small,
suggesting under-fitting (architecture / capacity bound) rather than
over-fitting. At this scale (~95k parameters), CIFAR-10 lives roughly in
the 60–70% accuracy range. The numbers match what the literature reports
for similarly sized CNNs without residual connections.

To push past 70%, the natural moves are *architectural* (deeper stack,
residual connections — Chapter 9) rather than more regularization. This
is the textbook case where capacity, not over-fitting, is the bottleneck.

### Reproducibility check

```
loaded checkpoint: epoch=9  reported val_acc=0.6515
re-evaluated val_loss=0.9988  val_acc=0.6515
```

`evaluate.py` re-loads `best.pt` and produces the same validation
accuracy bit-for-bit — no silent drift between the saved metric and the
metric on reload.

## Error analysis

For a 10-class problem at 65% accuracy, the typical confusions are the
ones you would expect from a small CNN on natural images:

- **bird ↔ deer / dog / cat** — small animals share textures and color
  palettes; the network has not learned to distinguish *shape* of the
  animal silhouette at this depth.
- **automobile ↔ truck** — the two classes are genuinely close (size and
  perspective dominate), and 32 × 32 hides distinguishing details.

A confusion matrix (generated from the saved checkpoint) would localize
these clusters precisely. Deeper architectures with residual connections
(Chapter 9) close most of these gaps.

## What I would try next

**One concrete experiment, with a hypothesis:** swap `SmallCNN` for a
ResNet-style 9-layer net with two `BasicBlock` stacks per stage. Same
training recipe, same epochs. Expected validation accuracy is **≥ 78%**
on CIFAR-10 — the residual connections both ease optimization and
double the effective receptive field per parameter spent. This is the
natural next step into Chapter 9.

If sticking with `SmallCNN`, the next ablation is a OneCycleLR schedule
(Chapter 7) — typical CIFAR-10 LR-warm-up + cosine-decay setups push
*this very architecture* a few points higher.

## Reproducibility

From the repo root:

```bash
pip install -r requirements.txt
python src/train.py     --config configs/cnn_cifar10.yaml
python src/evaluate.py  --config configs/cnn_cifar10.yaml \
                        --checkpoint experiments/cnn_cifar10/checkpoints/best.pt
```

Expected end state:

- `experiments/cnn_cifar10/checkpoints/best.pt` — best-by-val checkpoint
  (~380 KB, 94,762 float32 parameters).
- `experiments/cnn_cifar10/logs/history.json` — per-epoch train/val loss
  and accuracy.

With `seed: 42` pinned, the same machine reproduces 65.15% val accuracy.
