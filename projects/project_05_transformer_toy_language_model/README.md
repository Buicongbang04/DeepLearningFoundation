# Project 05 — Toy Transformer Language Model

A tiny decoder-only Transformer trained on a self-contained character-
level corpus. The fifth end-to-end project of the course, paired with
Chapter 12.

**Companion chapter:** `docs/11_transformer_intro.md`.

## The "char_corpus" dataset

A small Hamlet-soliloquy corpus baked into `src/_text_data.py` — no
external download. The corpus is repeated 16 times to give the model
enough sliding-window training material; vocabulary is the 25 unique
characters present (newline + space + lowercase letters).

This is intentionally a **memorization** dataset — small enough that a
~70k-parameter Transformer can fit and recite it. The point of the
project is to verify the **Transformer-LM training and sampling
pipeline** end-to-end, not to measure generalization.

## Reproduce

From the repo root:

```bash
pip install -r requirements.txt

python src/train_text.py --config configs/toy_transformer_charlm.yaml
```

Then sample text from the saved checkpoint via:

```python
import sys; sys.path.insert(0, "src")
import torch
from models import build_model

ckpt = torch.load("experiments/toy_transformer_charlm/checkpoints/best.pt",
                  map_location="cpu", weights_only=False)
itos = ckpt["itos"];  stoi = {c: i for i, c in enumerate(itos)}
model = build_model(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])

prompt = "to be or not"
ids = torch.tensor([[stoi[c] for c in prompt]])
out = model.generate(ids, max_new_tokens=80, temperature=0.8)[0].tolist()
print("".join(itos[i] for i in out))
```

## Configuration highlights

`configs/toy_transformer_charlm.yaml`:

- `data.name: char_corpus`, `block_size: 64`, `val_split: 0.1`.
- `model.name: char_transformer`, `d_model: 64`, `num_heads: 4`,
  `d_ff: 256`, `num_layers: 2`. About 73k parameters.
- `optim: adamw`, `lr: 3e-3`, `weight_decay: 1e-2`.
- `training: num_epochs: 10`, `clip_grad: 1.0`.

## Outputs

```
experiments/toy_transformer_charlm/
├── checkpoints/
│   ├── best.pt              # best-by-val-loss
│   └── last.pt
└── logs/
    └── history.json
```

Both directories are gitignored.

## Files

- `configs/toy_transformer_charlm.yaml` — single source of truth.
- `src/_text_data.py` — `_TINY_CORPUS` plus `char_corpus_loaders`.
- `src/datasets.py` — wraps the loader into the registry.
- `src/models.py` — `CharTransformerLM` (decoder-only, pre-norm, learned
  position embeddings) plus `_CausalEncoderBlock`.
- `src/train_text.py` — text-aware training driver. Handles the LM loss
  shape `(B*T, V)` automatically.
- `projects/project_05_transformer_toy_language_model/report.md` —
  written report with quantitative (loss, per-token acc) and
  qualitative (sampled completions) results.
