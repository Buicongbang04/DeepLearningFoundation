# Project 04 — LSTM Text Classifier

A small LSTM-based text classifier on a fully offline 4-class topic
dataset. The fourth end-to-end project of the course, paired with
Chapter 10.

**Companion chapter:** `docs/09_sequence_models.md`.

## The "synthetic_text" dataset

To keep the project completely reproducible without external network
downloads (no `torchtext`, no HuggingFace Hub), the dataset is built
*offline* from four small sentence-template pools — `weather`, `sports`,
`tech`, `food`. Each example is a paraphrase: a template with one or
two words randomly deleted. 200 train + 60 val examples per topic
(**800 train / 240 val**).

Vocabulary is built from the corpus itself; tokens are word-level with
`<pad>` and `<unk>`.

This is a *toy* dataset by design — the four topics are very easy to
separate, so the LSTM converges in one epoch. The point of the project
is **the pipeline** (Embedding → LSTM → Linear head, padded sequences,
checkpoint/eval flow), not benchmark accuracy.

## Reproduce

From the repo root:

```bash
pip install -r requirements.txt

python src/train_text.py    --config configs/lstm_text.yaml
python src/train_text.py    --config configs/lstm_text.yaml \
                            # evaluate.py is the same path; reload best.pt yourself
```

`train_text.py` is a sibling to `train.py`, dedicated to the text
dataloaders. The image projects continue to use `train.py`.

## Configuration highlights

`configs/lstm_text.yaml`:

- `data.name: synthetic_text`, 200 train + 60 val per topic, `max_len: 16`.
- `model.name: lstm_text`, `embed_dim: 64`, `hidden_dim: 128`,
  `num_layers: 1` — about 112k parameters.
- `optim: adam`, `lr: 5e-3`, gradient clipping at `1.0`.
- `training.num_epochs: 5`.

## Outputs

```
experiments/lstm_text/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── logs/
    └── history.json
```

Both directories are gitignored.

## Files

- `configs/lstm_text.yaml` — single source of truth.
- `src/_text_data.py` — synthetic-text and char-corpus factories.
- `src/datasets.py` — wraps `_text_data.py` into the registry.
- `src/models.py` — `LSTMTextClassifier` (`Embedding → LSTM → Dropout → Linear`).
- `src/train_text.py` — text-aware training driver (handles padded
  sequences and length-based packing).
- `projects/project_04_text_classification_lstm/report.md` — written
  report with the actual run numbers.
