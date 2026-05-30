"""Smoke tests for the loss + metric functions used in the training drivers.

The goal here is *not* to re-test PyTorch — it is to lock in the
specific shape conventions the repo's training loops depend on:

- `CrossEntropyLoss` expects raw logits `(B, C)` + integer labels `(B,)`.
- For the LM, logits are `(B, T, V)` and targets are `(B, T)`; the
  reshape in `train_text.loss_lm` collapses time into batch.
- Per-image MSE for the autoencoder is reduced *per example* (mean over
  pixel dims), then aggregated — the anomaly score depends on this.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- classification cross-entropy -----

def test_cross_entropy_returns_scalar():
    logits = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    loss = nn.CrossEntropyLoss()(logits, labels)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_cross_entropy_label_smoothing_returns_scalar():
    logits = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


# ----- LM loss reshape ----------------------------------------------------

def test_lm_loss_shape_collapse():
    """``train_text.loss_lm`` reshapes (B, T, V) → (B*T, V) and (B, T) → (B*T,).

    This is the shape convention every char-LM forward + cross-entropy
    in the repo depends on.
    """
    from train_text import loss_lm
    B, T, V = 3, 7, 9
    logits  = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    criterion = nn.CrossEntropyLoss()
    loss = loss_lm(logits, targets, criterion)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_lm_metric_returns_correct_count():
    from train_text import metric_lm
    B, T, V = 2, 4, 5
    targets = torch.tensor([[0, 1, 2, 3], [4, 0, 1, 2]])
    # Make the argmax always equal the target.
    logits = torch.full((B, T, V), -10.0)
    for b in range(B):
        for t in range(T):
            logits[b, t, targets[b, t]] = 10.0
    correct, total = metric_lm(logits, targets)
    assert total == B * T
    assert correct == B * T


def test_text_classifier_metric():
    from train_text import metric_text_classifier
    logits = torch.tensor([[0.1, 5.0], [5.0, 0.1], [0.1, 5.0]])
    targets = torch.tensor([1, 0, 0])
    correct, total = metric_text_classifier(logits, targets)
    assert total == 3
    assert correct == 2


# ----- per-image reconstruction MSE --------------------------------------

def test_per_image_mse_shape_and_reduction():
    """Project 06 scores per *image*; the reduction has to happen over
    the pixel dims, not the batch dim."""
    B, C, H, W = 5, 1, 28, 28
    x     = torch.zeros(B, C, H, W)
    x_hat = torch.full_like(x, 0.5)
    per   = (x_hat - x).pow(2).mean(dim=[1, 2, 3])
    assert per.shape == (B,)
    assert torch.allclose(per, torch.full((B,), 0.25))


def test_vae_loss_components_finite():
    """The VAE loss from the chapter_14 notebook: BCE + KL, normalized by batch."""
    B = 4
    x = torch.rand(B, 1, 28, 28)
    x_hat = torch.rand(B, 1, 28, 28).clamp(1e-6, 1 - 1e-6)
    mu     = torch.randn(B, 8)
    logvar = torch.randn(B, 8) * 0.1
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl    = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar).sum()
    loss  = (recon + kl) / B
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
