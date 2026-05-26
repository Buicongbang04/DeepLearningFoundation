# Chapter 6 — Regularization for Deep Learning

> A deep network has *more parameters than data points*. Without regularization, it will memorize the training set perfectly and learn nothing useful. This chapter is the toolkit for keeping the model from cheating.

## Mục tiêu (Goal)

After this chapter you can:

- Recognize overfitting from a training-vs-validation curve.
- Apply weight decay (L2), L1, dropout, and early stopping to a PyTorch model.
- Augment image data with `torchvision.transforms`.
- Explain noise injection, label smoothing, and ensembling at a conceptual level.

## Why this chapter

A deep model has so much capacity that it can hit 100% training accuracy on essentially any dataset, given enough epochs. That is *not* a sign of a good model — it is a sign of overfitting. Regularization is everything you do to push the model toward a *simpler* solution that generalizes.

- **Builds on:** Chapter 4 (training loop), Chapter 5 (activation/init/normalization).
- **Sets up:** Chapter 7 (optimization — AdamW couples nicely with weight decay), Chapter 9 (transfer learning needs dropout and weight decay), every project.

## Key concepts

### Overfitting in three pictures

Look at the training-vs-validation loss curve over epochs:

- **Under-fitting:** both losses high, both still decreasing. → train longer, add capacity.
- **Good fit:** both losses low; validation slightly above training. → keep this model.
- **Overfitting:** training loss keeps going down, validation loss bottoms out and goes back up. → regularize.

The gap between training and validation loss is the **generalization gap**. Regularization shrinks the gap.

### Weight decay (L2 regularization)

Add an `ℓ₂` penalty on the weights:

```
L_total = L_data + (λ / 2) · ‖θ‖²
```

This pulls every weight slightly toward zero each step. Small weights mean simpler functions, which generalize better.

In PyTorch:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
# or
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

Typical values: `1e-4` to `1e-2`. `AdamW` decouples weight decay from the adaptive learning rate — prefer it over `Adam(weight_decay=…)`.

### L1 regularization

Add an `ℓ₁` penalty instead:

```
L_total = L_data + λ · ‖θ‖₁
```

L1 encourages *sparsity* (many weights become exactly zero). Less common in deep learning than L2, but useful when you want a model that ignores most features.

PyTorch does not have an `--l1-decay` flag; you compute it explicitly:

```python
l1_lambda = 1e-5
l1_norm   = sum(p.abs().sum() for p in model.parameters())
loss      = data_loss + l1_lambda * l1_norm
```

### Dropout

During training, drop each neuron with probability `p`:

```python
self.fc1     = nn.Linear(in_dim, hidden_dim)
self.dropout = nn.Dropout(p=0.5)
self.fc2     = nn.Linear(hidden_dim, out_dim)

def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.dropout(x)
    return self.fc2(x)
```

What is happening:

- During training, each forward pass zeroes a random subset of activations.
- This forces the network to be *redundant* — no single unit can be relied on.
- During eval (`model.eval()`), dropout is *off* — every neuron is active and the outputs are scaled appropriately.

Typical `p` values:

- `0.5` after fully-connected layers in MLPs / CNN classification heads.
- `0.1`–`0.3` in Transformers and modern architectures.
- Almost never on convolutional feature maps directly (use BatchNorm + augmentation instead).

### Early stopping

Watch validation loss; stop training when it stops improving for `patience` epochs:

```python
best_val_loss = float("inf")
patience      = 5
epochs_no_improve = 0

for epoch in range(max_epochs):
    train_one_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        epochs_no_improve = 0
        save_checkpoint(...)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping.")
            break
```

Early stopping is *the* most reliable regularizer. Combine it with best-checkpoint saving so you always end up with the model from the best epoch.

### Data augmentation

Increasing dataset size by *transforming* training examples — `torchvision.transforms` for images:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(...),
])
```

Critical rules:

1. Augmentation **only** on the training set. Validation and test use a deterministic transform.
2. Augmentation must *not* change the label. Random rotation on a `6`/`9` digit example breaks MNIST. Horizontal flip on a "reading direction" task breaks NLP.
3. Strong augmentation is one of the cheapest wins in modern CV — `RandAugment`, `AutoAugment`, `Mixup`, `CutMix`.

For text:

- Synonym replacement, random insertion/swap/deletion.
- Back-translation.

For audio:

- SpecAugment (frequency and time masking).
- Speed and pitch perturbation.

### Noise injection

Add Gaussian noise to the input or to hidden activations:

```python
x_noisy = x + 0.1 * torch.randn_like(x)
```

A specific case is the **denoising autoencoder** (Chapter 14) — feed noisy inputs, reconstruct clean ones. Also useful as a generic regularizer.

### Label smoothing

Replace the one-hot label `[0, 0, 1, 0]` with `[0.025, 0.025, 0.925, 0.025]`:

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

This penalizes the model for being *too confident* and improves calibration. Common in image classification and machine translation.

### Ensembling (concept)

Train several models, average their predictions:

- **Bagging:** train each on a bootstrap sample.
- **Boosting:** train each to correct the previous one's errors (XGBoost, etc.).
- **Snapshot ensembling:** save checkpoints at multiple learning-rate minima of a cyclic schedule.

Ensembling almost always wins a Kaggle competition but doubles or triples inference cost. It is implicit in dropout (a single dropout-trained network behaves like an ensemble of sub-networks).

## A regularization recipe

For a brand-new image classifier:

1. **Augmentation first** — random crop, flip, color jitter. (cheapest win)
2. **Weight decay** in AdamW, `1e-2`.
3. **Dropout `p=0.5`** before the final classifier head.
4. **Early stopping** with `patience=5` on validation loss.
5. **Label smoothing `0.1`** if the model is over-confident.

Add ensembling only when the single model has been tuned.

## Common pitfalls

- Dropout `p=0.5` on convolutional layers → kills the spatial structure → use BN + augmentation instead.
- Forgot `model.eval()` during validation → dropout is still active → validation looks worse than it should.
- Augmenting the *validation* set → validation accuracy looks suspiciously noisy → reserve augmentation strictly for training.
- Adding weight decay to *BN parameters* (`γ, β`) → degrades performance → exclude them from `weight_decay` in your optimizer.
- Label smoothing `0.5` → confuses calibration with chaos → keep it in `[0.05, 0.15]`.

## Learning outcomes

- Read a learning curve and decide whether the model is under-fitting, well-fit, or over-fitting.
- Apply dropout and weight decay in PyTorch.
- Construct a `torchvision` augmentation pipeline that is safe for the label.
- Use `nn.CrossEntropyLoss(label_smoothing=0.1)` and explain what it does.

## Quick check (self-test)

<details>
<summary>Q1 — What does the training-vs-validation loss curve look like for a network that is over-fitting?</summary>

Training loss keeps going down; validation loss decreases, hits a minimum, and starts going up while training loss continues to drop. The gap between the two grows.
</details>

<details>
<summary>Q2 — Why is <code>AdamW</code> preferred over <code>Adam(weight_decay=…)</code> when you want L2 regularization?</summary>

`Adam(weight_decay)` adds the decay to the gradient before the adaptive learning rate, which couples decay to per-parameter learning rates and weakens it for frequently-updated parameters. `AdamW` decouples the weight-decay step from the gradient step, so each parameter is decayed by the same factor regardless of its history.
</details>

<details>
<summary>Q3 — You add dropout but training loss goes <em>down</em> faster instead of slower. What is the issue?</summary>

You probably forgot `model.eval()` during the validation loop, so the comparison is unfair (validation has dropout active). Or you placed dropout in the wrong spot. Verify by training without dropout and comparing the two curves on the same axes.
</details>

<details>
<summary>Q4 — Is random horizontal flip a safe augmentation for MNIST? Why or why not?</summary>

It is *not* safe for digit classification — flipping `2` produces `5`-ish, flipping `7` looks like garbage, and the *label has changed* visually but not in the dataset. Use random rotation in a small range and slight shifts instead.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 7 (regularization).
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014).
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, 2019).
- Müller et al., "When Does Label Smoothing Help?" (2019).
- PyTorch docs — `torch.nn.Dropout`, `torch.optim.AdamW`, `torchvision.transforms`.

## Companion artifact

`notebooks/chapter_05_regularization_dropout_batchnorm.ipynb` — overfitting demo, with-and-without dropout, weight-decay sweep, augmentation gallery.
