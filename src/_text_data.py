"""Helper module: text dataset factories for the synthetic_text and
char_corpus loaders. Imported by datasets.py to keep the main module
focused.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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
) * 16


def _build_word_vocab(sentences):
    PAD, UNK = "<pad>", "<unk>"
    words = sorted({w for s in sentences for w in s.split()})
    itos = [PAD, UNK] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def _encode(sentence, stoi, max_len):
    UNK = stoi["<unk>"]
    PAD = stoi["<pad>"]
    ids = [stoi.get(w, UNK) for w in sentence.split()][:max_len]
    length = len(ids)
    ids = ids + [PAD] * (max_len - length)
    return ids, length


class _SyntheticTextDataset(Dataset):
    def __init__(self, sentences, labels, lengths):
        self.x = torch.tensor(sentences, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.l = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.l[i], self.y[i]


def synthetic_text_loaders(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 0))
    n_train = int(data_cfg.get("examples_per_topic_train", 200))
    n_val   = int(data_cfg.get("examples_per_topic_val",   60))
    max_len = int(data_cfg.get("max_len", 16))

    topics = list(_SYNTHETIC_TEXT_TEMPLATES.keys())
    rng_train = np.random.default_rng(seed)
    rng_val   = np.random.default_rng(seed + 1)

    def _paraphrase(s, rng):
        words = s.split()
        if len(words) > 6 and rng.random() < 0.7:
            del words[int(rng.integers(0, len(words) - 1))]
        if len(words) > 6 and rng.random() < 0.4:
            del words[int(rng.integers(0, len(words) - 1))]
        return " ".join(words)

    train_sents, train_labels = [], []
    val_sents, val_labels = [], []
    for label_idx, topic in enumerate(topics):
        templates = _SYNTHETIC_TEXT_TEMPLATES[topic]
        for _ in range(n_train):
            t = templates[int(rng_train.integers(0, len(templates)))]
            train_sents.append(_paraphrase(t, rng_train))
            train_labels.append(label_idx)
        for _ in range(n_val):
            t = templates[int(rng_val.integers(0, len(templates)))]
            val_sents.append(_paraphrase(t, rng_val))
            val_labels.append(label_idx)

    stoi, itos = _build_word_vocab(train_sents + val_sents)

    def _encode_all(sents):
        ids_list, lens = [], []
        for s in sents:
            ids, length = _encode(s, stoi, max_len)
            ids_list.append(ids)
            lens.append(length)
        return ids_list, lens

    train_ids, train_lens = _encode_all(train_sents)
    val_ids, val_lens     = _encode_all(val_sents)

    train_ds = _SyntheticTextDataset(train_ids, train_labels, train_lens)
    val_ds   = _SyntheticTextDataset(val_ids,   val_labels,   val_lens)

    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                              shuffle=True, num_workers=data_cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=data_cfg["batch_size"],
                              shuffle=False, num_workers=data_cfg["num_workers"])
    info = {
        "num_classes": len(topics),
        "classes": topics,
        "vocab_size": len(itos),
        "stoi": stoi,
        "itos": itos,
        "max_len": max_len,
        "pad_id": stoi["<pad>"],
    }
    return train_loader, val_loader, info


class _CharCorpusDataset(Dataset):
    """Char-level next-token prediction dataset.

    Each example is a (block_size,) input and a (block_size,) target where
    target[i] = input[i+1]. Sliding-window over the whole corpus.
    """

    def __init__(self, ids: torch.Tensor, block_size: int):
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, i):
        chunk = self.ids[i : i + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def char_corpus_loaders(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    block_size = int(data_cfg.get("block_size", 64))
    val_split  = float(data_cfg.get("val_split", 0.1))

    chars = sorted(set(_TINY_CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = list(chars)
    ids = torch.tensor([stoi[c] for c in _TINY_CORPUS], dtype=torch.long)

    n = ids.size(0)
    n_val = int(n * val_split)
    n_train = n - n_val
    train_ids = ids[:n_train]
    val_ids   = ids[n_train:]

    train_ds = _CharCorpusDataset(train_ids, block_size)
    val_ds   = _CharCorpusDataset(val_ids,   block_size)

    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                              shuffle=True, num_workers=data_cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=data_cfg["batch_size"],
                              shuffle=False, num_workers=data_cfg["num_workers"])
    info = {
        "vocab_size": len(itos),
        "block_size": block_size,
        "stoi": stoi,
        "itos": itos,
    }
    return train_loader, val_loader, info
