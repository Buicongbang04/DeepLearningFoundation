"""Dataset / DataLoader factories for repo projects.

`build_loaders(config)` is the single entry point. It dispatches on
`config['data']['name']` and returns `(train_loader, val_loader, info)` where
``info`` is a small dict carrying things like ``num_classes`` and ``classes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _mnist_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((data_cfg["mean"],), (data_cfg["std"],)),
    ])

    train_ds = datasets.MNIST(root, train=True,  download=True, transform=transform)
    val_ds   = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    info = {
        "num_classes": 10,
        "classes": [str(i) for i in range(10)],
        "input_shape": (1, 28, 28),
    }
    return train_loader, val_loader, info


_REGISTRY = {
    "mnist": _mnist_loaders,
}


def build_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict]:
    name = config["data"]["name"]
    if name not in _REGISTRY:
        raise ValueError(f"unknown dataset: {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config)


def mnist_inference_transform(mean: float = 0.1307, std: float = 0.3081):
    """The exact preprocessing used at training time, for inference scripts."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])
