"""Model definitions for repo projects.

`build_model(config)` dispatches on ``config['model']['name']``. Every model
is a plain `nn.Module` taking the architecture hyperparameters as kwargs.
"""

from __future__ import annotations

from typing import Any

import math
import torch
import torch.nn as nn
from torchvision import models as tv_models


class MLPClassifier(nn.Module):
    """A small MLP for image classification.

    Inputs are flattened to ``in_dim`` before the first linear layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class SmallCNN(nn.Module):
    """Conv → BN → ReLU → Pool stack followed by an average-pool + Linear head.

    Default channel ramp (32 → 64 → 128) suits CIFAR-10. Pass
    ``in_channels=1`` for MNIST.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            _ConvBlock(in_channels, c1),
            _ConvBlock(c1, c2),
            _ConvBlock(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(config: dict[str, Any]) -> nn.Module:
    mcfg = config["model"]
    name = mcfg["name"]
    if name == "mlp":
        return MLPClassifier(
            in_dim=mcfg["in_dim"],
            hidden_dim=mcfg["hidden_dim"],
            num_classes=mcfg["num_classes"],
            dropout=mcfg.get("dropout", 0.0),
        )
    if name == "small_cnn":
        return SmallCNN(
            in_channels=mcfg.get("in_channels", 3),
            num_classes=mcfg["num_classes"],
            channels=tuple(mcfg.get("channels", (32, 64, 128))),
            dropout=mcfg.get("dropout", 0.0),
        )
    if name == "resnet18":
        return _build_resnet18(
            num_classes=mcfg["num_classes"],
            pretrained=mcfg.get("pretrained", True),
            freeze_backbone=mcfg.get("freeze_backbone", True),
        )
    if name == "lstm_text":
        return LSTMTextClassifier(
            vocab_size=mcfg["vocab_size"],
            embed_dim=mcfg.get("embed_dim", 64),
            hidden_dim=mcfg.get("hidden_dim", 128),
            num_classes=mcfg["num_classes"],
            num_layers=mcfg.get("num_layers", 1),
            dropout=mcfg.get("dropout", 0.0),
            pad_id=mcfg.get("pad_id", 0),
        )
    if name == "char_transformer":
        return CharTransformerLM(
            vocab_size=mcfg["vocab_size"],
            d_model=mcfg.get("d_model", 64),
            num_heads=mcfg.get("num_heads", 4),
            d_ff=mcfg.get("d_ff", 256),
            num_layers=mcfg.get("num_layers", 2),
            block_size=mcfg["block_size"],
            dropout=mcfg.get("dropout", 0.0),
        )
    if name == "conv_autoencoder":
        return ConvAutoencoder(
            in_channels=mcfg.get("in_channels", 1),
            latent_dim=mcfg.get("latent_dim", 32),
        )
    raise ValueError(f"unknown model: {name!r}")


class ConvAutoencoder(nn.Module):
    """A small symmetric conv autoencoder for 28x28 single-channel images.

    Encoder: 28x28 -> 14x14 -> 7x7 -> latent_dim.
    Decoder: latent_dim -> 7x7 -> 14x14 -> 28x28.
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 28 -> 14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),           # 14 -> 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7), nn.ReLU(inplace=True),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),  # 7 -> 14
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),                # 14 -> 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.0, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                             num_layers=num_layers,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc    = nn.Linear(hidden_dim, num_classes)
        self.pad_id = pad_id

    def forward(self, x, lengths=None):
        e = self.embed(x)                                # (B, T, E)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                e, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(e)
        last = h_n[-1]                                   # (B, hidden)
        return self.fc(self.dropout(last))


class _CausalEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        x_ln = self.ln1(x)
        a, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class CharTransformerLM(nn.Module):
    """Tiny decoder-only Transformer for char-level next-token prediction."""

    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256,
                 num_layers=2, block_size=128, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            _CausalEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.token_embed(ids) + self.pos_embed(pos)
        causal = torch.triu(torch.ones(T, T, device=ids.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, attn_mask=causal)
        x = self.ln_f(x)
        return self.head(x)               # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=128, temperature=1.0):
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            window = ids[:, -self.block_size:]
            logits = self.forward(window)[:, -1, :] / max(temperature, 1e-6)
            probs  = torch.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


def _build_resnet18(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """ResNet-18 with a replaced final classification head.

    When ``pretrained=True`` the backbone uses the torchvision default
    ImageNet weights. ``freeze_backbone=True`` keeps every parameter except
    the new ``fc`` head frozen, matching the "feature extraction" mode in
    docs/08_cnn_architectures_transfer_learning.md.
    """
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = tv_models.resnet18(weights=weights)

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    num_features = backbone.fc.in_features
    backbone.fc = nn.Linear(num_features, num_classes)   # requires_grad=True by default
    return backbone
