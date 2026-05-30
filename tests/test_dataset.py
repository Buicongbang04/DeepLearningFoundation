"""Smoke tests for every registered dataset loader.

Loaders that require an external download (mnist, cifar10,
cifar10_subset, fashion_mnist) are skipped when the corresponding
``datasets/<name>/`` folder is not present, so the test suite stays
green on a fresh checkout. The two offline loaders (`synthetic_text`,
`char_corpus`) are always exercised.
"""

import pytest
from pathlib import Path

from datasets import build_loaders


REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset_present(name: str, sentinel: str) -> bool:
    return (REPO_ROOT / "datasets" / name / sentinel).exists()


# ----- always-on (offline) loaders -----

def test_synthetic_text_shape():
    cfg = {
        "seed": 0,
        "data": {
            "name": "synthetic_text",
            "examples_per_topic_train": 16,
            "examples_per_topic_val": 8,
            "max_len": 12,
            "batch_size": 8,
            "num_workers": 0,
        },
    }
    train_loader, val_loader, info = build_loaders(cfg)
    assert info["num_classes"] == 4
    assert set(info["classes"]) == {"weather", "sports", "tech", "food"}
    assert info["vocab_size"] > 20
    x, lengths, y = next(iter(train_loader))
    assert x.shape == (8, 12)
    assert lengths.shape == (8,)
    assert y.shape == (8,)
    assert int(y.min()) >= 0 and int(y.max()) < info["num_classes"]


def test_char_corpus_shape():
    cfg = {
        "data": {
            "name": "char_corpus",
            "block_size": 16,
            "val_split": 0.1,
            "batch_size": 8,
            "num_workers": 0,
        },
    }
    train_loader, val_loader, info = build_loaders(cfg)
    assert info["vocab_size"] >= 25
    assert info["block_size"] == 16
    x, y = next(iter(train_loader))
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    # Targets are inputs shifted by one position, so they should not be
    # identical to the inputs (with high probability).
    assert not (x == y).all()


# ----- conditionally-on (download-required) loaders -----

@pytest.mark.skipif(not _dataset_present("mnist", "MNIST"),
                    reason="MNIST not yet downloaded to datasets/mnist/")
def test_mnist_shape():
    cfg = {
        "data": {
            "name": "mnist",
            "root": str(REPO_ROOT / "datasets" / "mnist"),
            "batch_size": 4,
            "num_workers": 0,
            "mean": 0.1307,
            "std": 0.3081,
        },
    }
    train_loader, _, info = build_loaders(cfg)
    assert info["num_classes"] == 10
    x, y = next(iter(train_loader))
    assert x.shape == (4, 1, 28, 28)
    assert y.shape == (4,)


@pytest.mark.skipif(not _dataset_present("cifar10", "cifar-10-batches-py"),
                    reason="CIFAR-10 not yet downloaded to datasets/cifar10/")
def test_cifar10_shape():
    cfg = {
        "data": {
            "name": "cifar10",
            "root": str(REPO_ROOT / "datasets" / "cifar10"),
            "batch_size": 4,
            "num_workers": 0,
            "augment": False,
        },
    }
    train_loader, _, info = build_loaders(cfg)
    assert info["num_classes"] == 10
    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, 32, 32)
    assert y.shape == (4,)


@pytest.mark.skipif(not _dataset_present("cifar10", "cifar-10-batches-py"),
                    reason="CIFAR-10 not yet downloaded — subset needs it")
def test_cifar10_subset_shape():
    cfg = {
        "data": {
            "name": "cifar10_subset",
            "root": str(REPO_ROOT / "datasets" / "cifar10"),
            "classes": ["airplane", "automobile", "bird", "cat"],
            "images_per_class_train": 8,
            "images_per_class_val": 4,
            "image_size": 64,
            "batch_size": 4,
            "num_workers": 0,
        },
    }
    train_loader, val_loader, info = build_loaders(cfg)
    assert info["num_classes"] == 4
    assert info["classes"] == ["airplane", "automobile", "bird", "cat"]
    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, 64, 64)
    assert y.shape == (4,)
    # Relabeled to 0..3.
    assert int(y.min()) >= 0 and int(y.max()) < 4


@pytest.mark.skipif(not _dataset_present("fashion_mnist", "FashionMNIST"),
                    reason="Fashion-MNIST not yet downloaded")
def test_fashion_mnist_shape():
    cfg = {
        "data": {
            "name": "fashion_mnist",
            "root": str(REPO_ROOT / "datasets" / "fashion_mnist"),
            "batch_size": 4,
            "num_workers": 0,
        },
    }
    train_loader, _, info = build_loaders(cfg)
    assert info["num_classes"] == 10
    x, y = next(iter(train_loader))
    assert x.shape == (4, 1, 28, 28)
    assert y.shape == (4,)
