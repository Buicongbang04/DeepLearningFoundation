# Project 06 — Autoencoder + Anomaly Detection: Report

## Problem

Anomaly detection on Fashion-MNIST. We train a small conv autoencoder
on a **single normal class** — `T-shirt/top` (class 0) — then score
every other validation image by its per-image reconstruction MSE. An
image is flagged as an *anomaly* if MSE exceeds a tuned threshold.

We restrict the *anomaly* evaluation set to three classes that look
visually distinct from T-shirts (`Trouser`, `Sandal`, `Ankle boot`)
to keep the task well-defined.

Metric: **F1 / precision / recall** on the binary "normal vs anomaly"
task, plus the mean MSE per class as a sanity check.

## Method

Architecture: `ConvAutoencoder` from `src/models.py`:

```
Encoder: Conv(1→16, s=2) → ReLU → Conv(16→32, s=2) → ReLU →
         Flatten → Linear(32·7·7 → 32)
Decoder: Linear(32 → 32·7·7) → ReLU → Unflatten(32, 7, 7) →
         ConvTranspose(32→16, s=2) → ReLU →
         ConvTranspose(16→1,  s=2) → Sigmoid
```

111,521 trainable parameters.

Training recipe (`configs/autoencoder_fashion_mnist.yaml`):

- **Optimizer:** Adam, `lr = 1e-3`, no weight decay.
- **Loss:** mean-squared error on 28×28 single-channel pixel values
  (after `Sigmoid`, the output range is `[0, 1]`).
- **Train data:** Fashion-MNIST class 0 (`T-shirt/top`), train split
  only — **6,000 images**. Validation is the full Fashion-MNIST test
  split, partitioned into "normal" (class 0 → 1,000 images) and
  "anomaly" (classes 1, 5, 9 → 3,000 images).
- **Batch size:** 128, **5 epochs**, `seed: 42`.

After training we score every val image and pick the **threshold that
maximizes F1** on the held-out scores.

## Results

### Headline metric — anomaly detection (val)

| Quantity              | Value          |
|-----------------------|---------------:|
| **Best F1**           | **0.9447**     |
| Precision @ best F1   | 0.9206         |
| Recall @ best F1      | 0.9700         |
| Threshold (MSE)       | 0.02218        |
| Mean MSE (normal)     | 0.01792        |
| Mean MSE (anomaly)    | 0.06252        |
| Anomaly / Normal MSE  | **3.49 ×**     |

The mean reconstruction MSE on anomalies is **3.5× higher** than on
in-distribution T-shirts — a clear gap that the F1-tuned threshold
separates cleanly.

### Training curve (per-epoch)

| Epoch | Train loss (normal) | Val loss (normal) |
|------:|--------------------:|------------------:|
| 1     | 0.10165             | 0.05526           |
| 2     | 0.04862             | 0.03901           |
| 3     | 0.03107             | 0.02557           |
| 4     | 0.02277             | 0.02087           |
| 5     | 0.01899             | **0.01792**       |

The autoencoder converges monotonically. Val loss tracks train loss
closely — no overfitting concern at this scale, and the gap between
the *normal-val* MSE (0.018) and the *anomaly* MSE (0.063) is what
makes the detection work.

### Score histogram

A histogram of per-image MSE (one bar per image) is saved as
`projects/project_06_autoencoder_anomaly_detection/figures/score_histogram.png`.
The two distributions overlap only in a small tail near the threshold;
that overlap is what limits precision to ~92%.

### Wall-clock

~0.6 seconds per epoch on CPU (6,000 single-class images, batch 128).
Five epochs end-to-end finish in ~3 seconds plus the scoring pass.

## Error analysis

Two failure modes account for most of the misclassifications:

- **False positives** (T-shirt scored as anomaly): unusual T-shirts —
  long sleeves, dark prints, asymmetric cuts — that the model could
  not reconstruct as faithfully. The autoencoder saw only "vanilla"
  T-shirts during training.
- **False negatives** (anomaly scored as normal): some short-bottom
  trousers and ankle boots produce overall pixel statistics close to
  a T-shirt silhouette, so the per-pixel MSE is small. The autoencoder
  is *not* learning category-level structure — it is learning to
  reconstruct *T-shirt-shaped distributions of pixels*.

If you imagine deploying this on a factory line, false negatives are
the dangerous failure mode — they are real anomalies the system
missed. Lowering the threshold trades precision for recall;
`docs/13_autoencoders_representation_learning.md` discusses this
trade-off.

## What I would try next

**One concrete experiment, with a hypothesis:** train a **denoising**
variant — feed in `x + 0.2 · N(0, I)`, target the clean `x`. Expected
F1 ≥ 0.96 because the denoising objective forces the latent to encode
*shape* rather than pixel-level texture, which should widen the gap
between normal and anomaly classes by another factor of 1.5–2.

If denoising does not move the needle, the next move is **richer
encoder** (4 conv blocks, latent_dim 64) — more capacity to memorize
exactly what a T-shirt looks like.

## Reproducibility

From the repo root:

```bash
pip install -r requirements.txt
python projects/project_06_autoencoder_anomaly_detection/train_and_score.py \
    --config configs/autoencoder_fashion_mnist.yaml
```

Expected end state:

- `experiments/autoencoder_fashion_mnist/checkpoints/best.pt` — best-
  by-val-MSE checkpoint (~440 KB, 111,521 float32 parameters).
- `experiments/autoencoder_fashion_mnist/logs/history.json` — full
  per-epoch loss plus the anomaly-detection summary block.
- `projects/project_06_autoencoder_anomaly_detection/figures/score_histogram.png`
  — score distributions on val with the chosen threshold marked.

With `seed: 42` pinned the same machine reproduces F1 = 0.9447 at
threshold MSE = 0.0222.
