"""Model definitions for repo projects.

`build_model(config)` dispatches on ``config['model']['name']``. Every model
is a plain `nn.Module` taking the architecture hyperparameters as kwargs.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


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
    raise ValueError(f"unknown model: {name!r}")
