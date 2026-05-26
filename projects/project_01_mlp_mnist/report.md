# Project 01 — MLP on MNIST: Report

## Problem

Multi-class digit classification on **MNIST** (10 classes, 60k train / 10k
test images, 28 × 28 grayscale). Metric: **top-1 validation accuracy** on the
official MNIST test set, with cross-entropy loss as the training objective.

The point of this project is the *training pipeline*, not state-of-the-art
accuracy. MNIST is the smallest benchmark where the standard PyTorch loop
(model, loss, optimizer, train/eval mode, best-checkpoint save) is worth
running once.

## Method

Single-hidden-layer MLP defined in `src/models.py` as `MLPClassifier`:

```
Flatten(28×28 → 784) → Linear(784 → 128) → ReLU → Linear(128 → 10)
```

101,770 trainable parameters. Trained with:

- **Optimizer:** Adam, `lr = 1e-3`, no weight decay.
- **Loss:** `nn.CrossEntropyLoss` on raw logits (no manual softmax).
- **Batch size:** 128.
- **Epochs:** 3.
- **Seed:** 42 (`random`, `numpy`, `torch`, `torch.cuda`).
- **Normalization:** MNIST channel mean = 0.1307, std = 0.3081.

The full configuration lives in `configs/mlp_mnist.yaml`.

## Results

### Headline metric

| Split        | Loss   | Accuracy |
|--------------|--------|----------|
| Train (ep 3) | 0.0909 | —        |
| **Val (ep 3, best)** | **0.0959** | **97.00%** |

### Training curve (per-epoch)

| Epoch | Train loss | Val loss | Val acc | Best? |
|------:|-----------:|---------:|--------:|:----:|
| 1     | 0.3009     | 0.1535   | 95.37%  | ✓    |
| 2     | 0.1335     | 0.1076   | 96.63%  | ✓    |
| 3     | 0.0909     | 0.0959   | 97.00%  | ✓    |

Train and validation loss decrease together over the 3 epochs — no sign of
overfitting yet at this scale. Best validation accuracy lands at the last
epoch; training longer with weight decay and dropout would likely push the
result a point or two higher (left as an extension below).

### Wall-clock

About **4.5 seconds per epoch on CPU** (no CUDA in this environment).
Three epochs end-to-end take well under a minute including MNIST download.

### Sanity-checked end-to-end pipeline

Five test-set digits were exported to PNGs and run through `inference.py`
to validate the full training-to-deployment path:

| File                       | True | Predicted | P(class) |
|----------------------------|:----:|:---------:|:--------:|
| sample_0_label7.png        | 7    | 7         | 0.999    |
| sample_1_label2.png        | 2    | 2         | 0.998    |
| sample_2_label1.png        | 1    | 1         | 0.994    |
| sample_3_label0.png        | 0    | 0         | 1.000    |
| sample_4_label4.png        | 4    | 4         | 0.999    |

All five correct, all with confidence ≥ 0.994. The inference script applies
the same `Normalize(0.1307, 0.3081)` transform as training (cf. Chapter 16).

### Reproducibility check

`python src/evaluate.py --config configs/mlp_mnist.yaml --checkpoint
experiments/mlp_mnist/checkpoints/best.pt` re-loads the best checkpoint and
re-computes the validation metric:

```
loaded checkpoint: epoch=2  reported val_acc=0.97
re-evaluated val_loss=0.0959  val_acc=0.9700
```

Numbers match the training-time report exactly — no drift, no leakage in
the checkpoint format.

## Error analysis

At 97% accuracy on 10k test images, roughly 300 examples are misclassified.
Two classes typically carry most of the errors on this MLP setup:

- **Class 4 ↔ 9** — long-tail similar pen-strokes.
- **Class 3 ↔ 5** — top-loop vs. open top.

Both confusions are well-known on shallow models without convolutional
inductive bias. A convolutional follow-up (Chapter 8) drops these
confusions sharply.

## What I would try next

**One concrete experiment, with a hypothesis:** replace the MLP with a
3-block CNN (Conv → BN → ReLU → Pool, twice, then a Linear head) at
matched parameter count, train with the same Adam + LR for 5 epochs.
Expected validation accuracy is **≥ 99.0%** on MNIST, based on the LeNet
literature. This is the natural next step into Chapter 8.

If MLPs are still the focus, the next ablation is dropout 0.2 in the hidden
layer + weight decay `5e-4`; this should push val acc from 97.0% to about
97.5–98.0% before the architecture itself becomes the bottleneck.

## Reproducibility

From the repo root:

```bash
pip install -r requirements.txt

# Train.
python src/train.py     --config configs/mlp_mnist.yaml

# Re-evaluate the best checkpoint.
python src/evaluate.py  --config configs/mlp_mnist.yaml \
                        --checkpoint experiments/mlp_mnist/checkpoints/best.pt

# Predict on a single 28x28 image or a directory of images.
python src/inference.py --checkpoint experiments/mlp_mnist/checkpoints/best.pt \
                        --input <png-or-directory>
```

Expected end state:

- `experiments/mlp_mnist/checkpoints/best.pt` — best-by-val checkpoint
  (~410 KB, 101,770 float32 parameters).
- `experiments/mlp_mnist/checkpoints/last.pt` — checkpoint at final epoch.
- `experiments/mlp_mnist/logs/history.json` — per-epoch train/val
  loss and accuracy plus the config that produced them.

With `seed: 42` pinned, the same machine reproduces 97.00% val accuracy
bit-identically.
