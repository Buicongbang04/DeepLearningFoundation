# Chapter 4 — A Standard Training Loop in PyTorch

> The PyTorch training loop is the most copy-pasted snippet in Deep Learning. Type it out from memory five times. After that, you can spend your attention on the model and the data, not on the boilerplate.

## Mục tiêu (Goal)

After this chapter you can:

- Build an `nn.Module`, an `Optimizer`, a loss function, a `Dataset`, and a `DataLoader`.
- Write the canonical training loop with `model.train()`, validation under `model.eval() + no_grad`, best-checkpoint saving, and per-epoch logging.
- Load a saved checkpoint and run inference on a single example.
- Split the code into `models.py`, `train.py`, `evaluate.py`, `inference.py`.

## Why this chapter

This chapter is where the course flips from *understanding* to *engineering*. You will write essentially the same loop in every later chapter — for CNNs, RNNs, Transformers, autoencoders. The only things that change are the model, the loss, and the dataset.

- **Builds on:** Chapter 1 (tensor, autograd), Chapter 2-3 (MLP, backprop, autograd cross-check).
- **Sets up:** every project from Chapter 4 onward.

## Key concepts

### `Dataset` and `DataLoader`

A `Dataset` is the thing that knows how to return *one* example by index. A `DataLoader` is the thing that batches, shuffles, and parallelizes loading. The split is deliberate: a `Dataset` is small, fast, and unit-testable; a `DataLoader` is the I/O machine.

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean/std
])

train_ds = datasets.MNIST("datasets", train=True,  download=True, transform=transform)
val_ds   = datasets.MNIST("datasets", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)
```

A custom dataset implements three methods:

```python
class MyDataset(Dataset):
    def __init__(self, …):
        ...
    def __len__(self):
        return self._N
    def __getitem__(self, idx):
        return x_i, y_i
```

### `nn.Module` — the model

Every model inherits from `nn.Module`:

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

Two rules:

1. *Define* layers in `__init__`. Calling them in `forward` registers them as parameters.
2. Do *not* call `.cuda()` or `.to(...)` inside `__init__`. Do it once on the outside, after construction.

### Loss and optimizer

```python
import torch
import torch.nn.functional as F
from torch import optim

model     = MLP(28 * 28, 128, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

`CrossEntropyLoss` expects *raw logits* (no softmax) and integer labels (not one-hot). It applies `log_softmax` internally for numerical stability.

### The standard training loop

```python
for epoch in range(num_epochs):
    # ------- Train -------
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    # ------- Validate -------
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    val_loss /= len(val_loader.dataset)
    val_acc   = correct / total

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
```

Memorize the six-line core:

```python
optimizer.zero_grad()
outputs = model(images)
loss    = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

Every later loop is a variation on this.

### Best-checkpoint saving

Save the *best* model by validation metric, not the last one:

```python
best_val_acc = 0.0
ckpt_path    = "experiments/checkpoints/best.pt"

if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }, ckpt_path)
```

`model.state_dict()` is a `dict` from parameter name to tensor. Saving the dict is more flexible than saving the whole `model` object (which is sensitive to class-path changes).

### Loading a checkpoint for inference

```python
ckpt  = torch.load("experiments/checkpoints/best.pt", map_location=device)
model = MLP(28 * 28, 128, 10).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
```

You instantiate the model first, then load the weights into it.

### Random seed

Every project should pin a random seed at the top:

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

This makes runs reproducible. It does *not* guarantee bit-identical results across GPU types or driver versions, but it makes "the same script gives the same number twice" true on the same machine.

### Splitting code into files

A clean project layout:

```
src/
├── datasets.py    # build train_ds, val_ds
├── models.py      # define MLP
├── train.py       # parses --config, runs the training loop
├── evaluate.py    # parses --config + --checkpoint, runs eval-only
└── inference.py   # parses --checkpoint + --input, runs single-example predict
```

The CLI looks like:

```bash
python src/train.py     --config configs/mlp_mnist.yaml
python src/evaluate.py  --config configs/mlp_mnist.yaml --checkpoint experiments/checkpoints/best.pt
python src/inference.py --checkpoint experiments/checkpoints/best.pt --input sample.png
```

You should be able to give someone *only* the config and the checkpoint and they can reproduce the metrics. This is the reproducibility standard from `CONTRIBUTING.md`.

## Common pitfalls

- Forgot `optimizer.zero_grad()` → gradients accumulate, loss grows or oscillates. Put `zero_grad()` as the very first line of the training step.
- Forgot `model.eval()` during validation → Dropout still randomly zeroes activations, BatchNorm uses batch stats → metrics look noisy and worse than reality.
- Forgot `torch.no_grad()` during validation → memory creeps up, eval is slow. Add it.
- Saving `model` (the whole object) instead of `model.state_dict()` → load fails after refactor. Save the dict.
- Loss does not decrease → 90% of the time it is the learning rate. Sweep `[1e-2, 1e-3, 1e-4]` before changing anything else.
- Val accuracy is 100% from epoch 1 → almost certainly a *data leak*: you used the same data for both train and val by accident.

## Learning outcomes

- Build a complete PyTorch pipeline (data, model, loss, optimizer, train/val, checkpoint).
- Save the best checkpoint by validation accuracy and reload it for inference.
- Split your project into `train.py`, `evaluate.py`, `inference.py`, `models.py`, `datasets.py`.
- Train an MLP on MNIST to ≥ 97% validation accuracy in under 5 epochs on CPU.

## Quick check (self-test)

<details>
<summary>Q1 — What are the five lines of the inner training step, in order?</summary>

`optimizer.zero_grad()` → `outputs = model(x)` → `loss = criterion(outputs, y)` → `loss.backward()` → `optimizer.step()`.
</details>

<details>
<summary>Q2 — During validation you need to switch <em>two</em> things off. What are they and what does each one do?</summary>

`model.eval()` switches off Dropout (no random zeroing) and switches BatchNorm to use running stats. `torch.no_grad()` switches off the autograd graph so no memory is wasted and no gradients are tracked.
</details>

<details>
<summary>Q3 — Should you save <code>torch.save(model, ...)</code> or <code>torch.save(model.state_dict(), ...)</code> and why?</summary>

Save the `state_dict`. It is a plain dictionary of parameter tensors and survives refactors of the model class. Saving the whole object hard-codes the import path.
</details>

<details>
<summary>Q4 — Your model trains, but every epoch the validation accuracy is exactly the same as the training accuracy. What do you check first?</summary>

A data leak — the validation set is probably a copy of the training set. Check the split, the indices, the file paths, and any caching layer.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 6 (forward pass), Chapter 8 (optimization).
- PyTorch official tutorial — *Save and Load the Model*, *Training a Classifier*.
- The PyTorch *Datasets & DataLoaders* tutorial — covers custom `Dataset` and `DataLoader` worker pitfalls.

## Companion artifact

`projects/project_01_mlp_mnist/` — the MLP-on-MNIST project, full layout, with `configs/mlp_mnist.yaml`, `src/train.py`, `src/evaluate.py`, `src/inference.py`.
