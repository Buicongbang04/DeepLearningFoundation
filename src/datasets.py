"""Dataset / DataLoader factories for repo projects.

`build_loaders(config)` is the single entry point. It dispatches on
`config['data']['name']` and returns `(train_loader, val_loader, info)` where
``info`` is a small dict carrying things like ``num_classes`` and ``classes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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


def _cifar10_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    root.mkdir(parents=True, exist_ok=True)

    mean = tuple(data_cfg.get("mean", (0.4914, 0.4822, 0.4465)))
    std  = tuple(data_cfg.get("std",  (0.2470, 0.2435, 0.2616)))

    train_transform_list: list = []
    if data_cfg.get("augment", True):
        train_transform_list += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    train_transform_list += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_transform = transforms.Compose(train_transform_list)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root, train=True,  download=True, transform=train_transform)
    val_ds   = datasets.CIFAR10(root, train=False, download=True, transform=val_transform)

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
        "classes": list(train_ds.classes),
        "input_shape": (3, 32, 32),
    }
    return train_loader, val_loader, info


def _cifar10_subset_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict]:
    """A *small custom* dataset built from a few CIFAR-10 classes, upscaled to
    ImageNet resolution for transfer learning.

    Config knobs (under ``data:``):
      - ``classes`` : list of CIFAR-10 class names to keep (e.g.
        ``["airplane", "automobile", "bird", "cat"]``).
      - ``images_per_class_train`` and ``images_per_class_val``: subsample sizes.
      - ``image_size`` (default 224): square resize.
    """
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    root.mkdir(parents=True, exist_ok=True)

    classes_keep: list[str] = list(data_cfg["classes"])
    n_train = int(data_cfg.get("images_per_class_train", 200))
    n_val   = int(data_cfg.get("images_per_class_val",   100))
    image_size = int(data_cfg.get("image_size", 224))

    # ImageNet stats: required when feeding a pretrained backbone.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_full = datasets.CIFAR10(root, train=True,  download=True, transform=train_transform)
    val_full   = datasets.CIFAR10(root, train=False, download=True, transform=val_transform)

    name_to_idx = {name: i for i, name in enumerate(train_full.classes)}
    keep_indices = [name_to_idx[c] for c in classes_keep]
    keep_set = set(keep_indices)
    remap = {orig: new for new, orig in enumerate(keep_indices)}

    def _build_subset(ds, max_per_class):
        per_class_count = {orig: 0 for orig in keep_indices}
        picked = []
        for i, target in enumerate(ds.targets):
            if target in keep_set and per_class_count[target] < max_per_class:
                picked.append(i)
                per_class_count[target] += 1
        return picked

    train_idx = _build_subset(train_full, n_train)
    val_idx   = _build_subset(val_full,   n_val)

    class _RelabeledSubset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            x, y = self.base[self.indices[i]]
            return x, remap[y]

    train_ds = _RelabeledSubset(train_full, train_idx)
    val_ds   = _RelabeledSubset(val_full,   val_idx)

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
        "num_classes": len(classes_keep),
        "classes": classes_keep,
        "input_shape": (3, image_size, image_size),
    }
    return train_loader, val_loader, info


def _synthetic_text_loaders(cfg):
    from _text_data import synthetic_text_loaders
    return synthetic_text_loaders(cfg)


def _char_corpus_loaders(cfg):
    from _text_data import char_corpus_loaders
    return char_corpus_loaders(cfg)


def _fashion_mnist_loaders(cfg):
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(root, train=True,  download=True, transform=transform)
    val_ds   = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=data_cfg["batch_size"], shuffle=False,
                              num_workers=data_cfg["num_workers"])
    info = {
        "num_classes": 10,
        "classes": list(train_ds.classes),
        "input_shape": (1, 28, 28),
    }
    return train_loader, val_loader, info


_REGISTRY = {
    "mnist":          _mnist_loaders,
    "cifar10":        _cifar10_loaders,
    "cifar10_subset": _cifar10_subset_loaders,
    "synthetic_text": _synthetic_text_loaders,
    "char_corpus":    _char_corpus_loaders,
    "fashion_mnist":  _fashion_mnist_loaders,
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


# ----------------------------------------------------------------------
# Text datasets — fully self-contained (no network downloads).
# ----------------------------------------------------------------------

# Five small sentence templates per topic. The classifier task is to map a
# sentence back to its topic. Vocabulary is built from the corpus itself.
_SYNTHETIC_TEXT_TEMPLATES: dict[str, list[str]] = {
    "weather": [
        "the weather is sunny and warm today",
        "heavy rain will fall this afternoon and evening",
        "snow is expected in the mountains tonight",
        "winds will be strong near the coast tomorrow",
        "clouds are clearing after the morning storm",
        "a cold front is moving across the valley",
        "thunderstorms may arrive in the late afternoon",
        "temperatures will drop sharply by sunset",
    ],
    "sports": [
        "the team scored a goal in the final minute",
        "the home crowd cheered after the winning shot",
        "the coach selected three new players for the squad",
        "the match ended in a tie after extra time",
        "the striker missed an open goal early in the half",
        "the defender blocked the shot near the goal line",
        "the referee awarded a penalty for the foul",
        "the captain lifted the trophy after the final whistle",
    ],
    "tech": [
        "the new chip improves battery life by twenty percent",
        "the company released a new operating system update",
        "the cloud platform now supports faster machine learning training",
        "the open source library was downloaded one million times",
        "the developers fixed a critical security bug in the framework",
        "the model uses transformer layers and trains end to end",
        "the api adds support for streaming json responses",
        "the package manager resolves conflicts much faster now",
    ],
    "food": [
        "the restaurant serves fresh pasta with seasonal vegetables",
        "the chef prepared a simple soup with herbs and bread",
        "the bakery opens early and sells out by noon",
        "the dish combines rice fish and a sweet sauce",
        "the cafe offers coffee tea and homemade pastries",
        "the salad uses tomatoes lettuce cheese and olive oil",
        "the dessert balances chocolate fruit and a touch of salt",
        "the menu changes every week with local produce",
    ],
}


def _build_word_vocab(sentences: list[str]) -> tuple[dict[str, int], list[str]]:
    PAD, UNK = "<pad>", "<unk>"
    words = sorted({w for s in sentences for w in s.split()})
    itos = [PAD, UNK] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def _encode(sentence: str, stoi: dict[str, int], max_len: int) -> tuple[list[int], int]:
    UNK = stoi["<unk>"]
    PAD = stoi["<pad>"]
    ids = [stoi.get(w, UNK) for w in sentence.split()][:max_len]
    length = len(ids)
    ids = ids + [PAD] * (max_len - length)
    return ids, length


class _SyntheticTextDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, lengths):
        self.x = torch.tensor(sentences, dtype=torch.long)
        self.y = torch.tensor(labels,    dtype=torch.long)
        self.l = torch.tensor(lengths,   dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.x[i], self.l[i], self.y[i]


def _synthetic_text_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict]:
    """Synthetic 4-class topic-classification dataset.

    For each topic, we paraphrase its templates by randomly *deleting*
    one or two adjacent words to vary the input. Train and val use
    different paraphrase seeds so the val set is not memorizable.
    """
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 0))
    n_train_per_topic = int(data_cfg.get("examples_per_topic_train", 200))
    n_val_per_topic   = int(data_cfg.get("examples_per_topic_val",   60))
    max_len = int(data_cfg.get("max_len", 16))

    topics = list(_SYNTHETIC_TEXT_TEMPLATES.keys())

    rng_train = np.random.default_rng(seed)
    rng_val   = np.random.default_rng(seed + 1)

    def _paraphrase(s: str, rng) -> str:
        words = s.split()
        if len(words) > 6 and rng.random() < 0.7:
            i = int(rng.integers(0, len(words) - 1))
            del words[i]
        if len(words) > 6 and rng.random() < 0.4:
            i = int(rng.integers(0, len(words) - 1))
            del words[i]
        return " ".join(words)

    train_sents, train_labels = [], []
    val_sents,   val_labels   = [], []
    for label_idx, topic in enumerate(topics):
        templates = _SYNTHETIC_TEXT_TEMPLATES[topic]
        for _ in range(n_train_per_topic):
            t = templates[int(rng_train.integers(0, len(templates)))]
            train_sents.append(_paraphrase(t, rng_train))
            train_labels.append(label_idx)
        for _ in range(n_val_per_topic):
            t = templates[int(rng_val.integers(0, len(templates)))]
            val_sents.append(_paraphrase(t, rng_val))
            val_labels.append(label_idx)

    stoi, itos = _build_word_vocab(train_sents + val_sents)

    def _encode_all(sents):
        ids_list, lens = [], []
        for s in sents:
            ids, length = _encode(s, stoi, max_len)
            ids_list.append(ids); lens.append(length)
        return ids_list, lens

    train_ids, train_lens = _encode_all(train_sents)
    val_ids,   val_lens   = _encode_all(val_sents)

    train_ds = _SyntheticTextDataset(train_ids, train_labels, train_lens)
    val_ds   = _SyntheticTextDataset(val_ids,   val_labels,   val_lens)

    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                              shuffle=True,  num_workers=data_cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=data_cfg["batch_size"],
                              shuffle=False, num_workers=data_cfg["num_workers"])
    info = {
        "num_classes": len(topics),
        "classes": topics,
        "vocab_size": len(itos),
        "pad_id": stoi["<pad>"],
        "unk_id": stoi["<unk>"],
        "max_len": max_len,
        "stoi": stoi,
        "itos": itos,
    }
    return train_loader, val_loader, info


# ----------------------------------------------------------------------
# Char-level corpus for the toy decoder-only Transformer LM.
# ----------------------------------------------------------------------

_CHAR_CORPUS = (
    "to be or not to be that is the question. "
    "whether tis nobler in the mind to suffer the slings and arrows of "
    "outrageous fortune or to take arms against a sea of troubles. "
    "and by opposing end them. to die to sleep no more. and by a sleep "
    "to say we end the heart-ache and the thousand natural shocks that "
    "flesh is heir to. tis a consummation devoutly to be wished. "
    "to die to sleep. to sleep perchance to dream ay there is the rub. "
    "for in that sleep of death what dreams may come when we have "
    "shuffled off this mortal coil must give us pause. "
) * 6   # ~5kB of text — small enough for CPU; large enough for a 4-layer toy LM.


class _CharLMDataset(torch.utils.data.Dataset):
    """Sliding-window char-level dataset: x is chars[i:i+block], y is shifted by one."""

    def __init__(self, ids: list[int], block_size: int):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, i):
        x = self.ids[i:i + self.block_size]
        y = self.ids[i + 1:i + 1 + self.block_size]
        return x, y


def _char_corpus_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict]:
    data_cfg = cfg["data"]
    block_size = int(data_cfg.get("block_size", 64))

    text = _CHAR_CORPUS
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = chars
    ids = [stoi[c] for c in text]

    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]

    train_ds = _CharLMDataset(train_ids, block_size)
    val_ds   = _CharLMDataset(val_ids,   block_size)
    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                              shuffle=True, num_workers=data_cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=data_cfg["batch_size"],
                              shuffle=False, num_workers=data_cfg["num_workers"])
    info = {
        "vocab_size": len(itos),
        "block_size": block_size,
        "stoi": stoi,
        "itos": itos,
        "num_classes": len(itos),       # for the LM head
    }
    return train_loader, val_loader, info

_SYNTHETIC_TEXT_TEMPLATES: dict[str, list[str]] = {
    "weather": [
        "the weather is sunny and warm today",
        "heavy rain will fall this afternoon and evening",
        "snow is expected in the mountains tonight",
        "winds will be strong near the coast tomorrow",
        "clouds are clearing after the morning storm",
        "a cold front is moving across the valley",
        "thunderstorms may arrive in the late afternoon",
        "temperatures will drop sharply by sunset",
    ],
    "sports": [
        "the team scored a goal in the final minute",
        "the home crowd cheered after the winning shot",
        "the coach selected three new players for the squad",
        "the match ended in a tie after extra time",
        "the striker missed an open goal early in the half",
        "the defender blocked the shot near the goal line",
        "the referee awarded a penalty for the foul",
        "the captain lifted the trophy after the final whistle",
    ],
    "tech": [
        "the new chip improves battery life by twenty percent",
        "the company released a new operating system update",
        "the cloud platform now supports faster machine learning training",
        "the open source library was downloaded one million times",
        "the developers fixed a critical security bug in the framework",
        "the model uses transformer layers and trains end to end",
        "the api adds support for streaming json responses",
        "the package manager resolves conflicts much faster now",
    ],
    "food": [
        "the restaurant serves fresh pasta with seasonal vegetables",
        "the chef prepared a simple soup with herbs and bread",
        "the bakery opens early and sells out by noon",
        "the dish combines rice fish and a sweet sauce",
        "the cafe offers coffee tea and homemade pastries",
        "the salad uses tomatoes lettuce cheese and olive oil",
        "the dessert balances chocolate fruit and a touch of salt",
        "the menu changes every week with local produce",
    ],
}


_TINY_CORPUS = (
    "to be or not to be that is the question\n"
    "whether tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune\n"
    "or to take arms against a sea of troubles\n"
    "and by opposing end them to die to sleep\n"
    "no more and by a sleep to say we end\n"
    "the heart ache and the thousand natural shocks\n"
    "that flesh is heir to tis a consummation\n"
    "devoutly to be wished to die to sleep\n"
    "to sleep perchance to dream ay theres the rub\n"
) * 12  # repeat to give enough material to train on


def _build_word_vocab(sentences: list[str]) -> tuple[dict[str, int], list[str]]:
    PAD, UNK = "<pad>", "<unk>"
    words = sorted({w for s in sentences for w in s.split()})
    itos = [PAD, UNK] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def _encode(sentence: str, stoi: dict[str, int], max_len: int) -> tuple[list[int], int]:
    UNK = stoi["<unk>"]
    PAD = stoi["<pad>"]
    ids = [stoi.get(w, UNK) for w in sentence.split()][:max_len]
    length = len(ids)
    ids = ids + [PAD] * (max_len - length)
    return ids, length


class _SyntheticTextDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, lengths):
        self.x = torch.tensor(sentences, dtype=torch.long)
        self.y = torch.tensor(labels,    dtype=torch.long)
        self.l = torch.tensor(lengths,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.l[i], self.y[i]


def _synthetic_text_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict]:
    """Synthetic 4-class topic-classification dataset (offline, no downloads)."""
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 0))
    n_train_per_topic = int(data_cfg.get("examples_per_topic_train", 200))
    n_val_per_topic   = int(data_cfg.get("examples_per_topic_val",   60))
    max_len = int(data_cfg.get("max_len", 16))

    topics = list(_SYNTHETIC_TEXT_TEMPLATES.keys())
    rng_train = np.random.default_rng(seed)
    rng_val   = np.random.default_rng(seed + 1)

    def _paraphrase(s: str, rng) -> str:
        words = s.split()
        if len(words) > 6 and rng.random() < 0.7:
            i = int(rng.integers(0, len(words) - 1))
            del words[i]
        if len(words) > 6 and rng.random() < 0.4:
            i = int(rng.integers(0, len(words) - 1))
            del words[i]
        return " ".join(words)

    train_sents, train_labels = [], []
    val_sents,   val_labels   = [], []
    for label_idx, topic in enumerate(topics):
        templates = _SYNTHETIC_TEXT_TEMPLATES[topic]
        for _ in range(n_train_per_topic):
            t = templates[int(rng_train.integers(0, len(templates)))]
            train_sents.append(_paraphrase(t, rng_train))
            train_labels.append(label_idx)
        for _ in range(n_val_per_topic):
            t = templates[int(rng_val.integers(0, len(templates)))]
            val_sents.append(_paraphrase(t, rng_val))
            val_labels.append(label_idx)

    stoi, itos = _build_word_vocab(train_sents + val_sents)

    def _encode_all(sents):
        ids_list, lens = [], []
        for s in sents:
            ids, length = _encode(s, stoi, max_len)
            ids_list.append(ids); lens.append(length)
        return ids_list, lens

    train_ids, train_lens = _encode_all(train_sents)
    val_ids,   val_lens   = _encode_all(val_sents)

    train_ds = _SyntheticTextDataset(train_ids, train_labels, train_lens)
    val_ds   = _SyntheticTextDataset(val_ids,   val_lens,   val_labels)

    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=data_cfg["batch_size"], shuffle=False,
                              num_workers=data_cfg["num_workers"])

    info = {
        "num_classes": len(topics),
        "classes": topics,
        "vocab_size": len(itos),
        "stoi": stoi,
        "itos": itos,
        "max_len": max_len,
    }
    return train_loader, val_loader, info


class _CharBlockDataset(torch.utils.data.Dataset):
    """Sliding-window dataset over a character corpus.

    Each example is ``(x, y)`` where ``x = data[i:i+block]`` and
    ``y = data[i+1:i+block+1]`` — next-character prediction at every
    position in the block.
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return self.data.size(0) - self.block_size - 1

    def __getitem__(self, i):
        x = self.data[i:i + self.block_size]
        y = self.data[i + 1:i + 1 + self.block_size]
        return x, y


def _char_corpus_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict]:
    """Tiny character-level corpus for the toy Transformer language model.

    The corpus is bundled in this file (no downloads). Train/val split is
    a 90/10 contiguous split of the character stream.
    """
    data_cfg = cfg["data"]
    block_size = int(data_cfg.get("block_size", 64))
    text = data_cfg.get("text") or _TINY_CORPUS

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = chars
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = encoded.size(0)
    split = int(n * 0.9)
    train_data = encoded[:split]
    val_data   = encoded[split:]

    train_ds = _CharBlockDataset(train_data, block_size)
    val_ds   = _CharBlockDataset(val_data,   block_size)

    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=data_cfg["batch_size"], shuffle=False,
                              num_workers=data_cfg["num_workers"])

    info = {
        "vocab_size": len(itos),
        "stoi": stoi,
        "itos": itos,
        "block_size": block_size,
        "corpus_chars": n,
    }
    return train_loader, val_loader, info
