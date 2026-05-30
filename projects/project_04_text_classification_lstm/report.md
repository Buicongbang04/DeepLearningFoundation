# Project 04 — LSTM Text Classifier: Report

## Problem

4-class topic classification on a fully offline synthetic dataset built
from 32 sentence templates (8 per topic). Train: 800 sentences, val:
240 sentences. Metric: **validation accuracy** plus cross-entropy loss.

Because the dataset is synthetic-by-design (no external download
dependency, no `torchtext`, no HuggingFace Hub), the project's value is
the **end-to-end text pipeline**, not the headline accuracy. The
templates are deliberately separable enough that an LSTM converges in
one epoch — exactly what you want from a Chapter 10 walking-skeleton
project.

## Method

Architecture: `LSTMTextClassifier` from `src/models.py`:

```
Embedding(vocab=186, dim=64, padding_idx=0)
LSTM(input=64, hidden=128, num_layers=1, batch_first=True)
Linear(128 → 4)
```

111,748 trainable parameters.

Training recipe (`configs/lstm_text.yaml`):

- **Optimizer:** Adam, `lr = 5e-3`, no weight decay.
- **Gradient clipping:** `clip_grad_norm_(1.0)` — standard for LSTMs.
- **Padded sequences:** `pack_padded_sequence(enforce_sorted=False)` so
  the LSTM only sees real tokens, not pad positions.
- **Batch size:** 32, **5 epochs**, `seed: 42`.

## Results

### Headline metric

| Epoch | Train loss | Val loss | Val acc |
|------:|-----------:|---------:|--------:|
| 1     | 0.3920     | 0.0046   | 100.00% |
| 2     | 0.0011     | 0.0003   | 100.00% |
| 3     | 0.0002     | 0.0001   | 100.00% |
| 4     | 0.0001     | 0.0001   | 100.00% |
| 5     | 0.0001     | 0.0001   | 100.00% |

**Best val acc: 100.00% at epoch 5** (any epoch ≥ 1 already hits 100%).

### Wall-clock

~0.1 seconds per epoch on CPU. Five epochs end-to-end finish in under
two seconds (the dominant cost is the eval pass, not training).

### Interpretation

The four topics each have characteristic vocabulary (`goal`, `striker`
for sports; `chip`, `transformer` for tech; `pasta`, `bakery` for food;
`rain`, `clouds` for weather). The LSTM only needs to learn that
*presence of one topic word* is enough — the task is trivially
linearly-separable in the embedding mean. We are seeing the LSTM
exploit that.

This is the chapter's **walking-skeleton** result: every component of
the text pipeline (vocab, padding, packing, length-aware forward,
classification head, eval) works correctly. Real-world text (AG News,
IMDB) requires the dependency to be available; the README documents
the upgrade path.

## What I would try next

**One concrete experiment, with a hypothesis:** feed the same model
real text — AG News (4 classes, ~120k train samples) via
`pip install datasets && load_dataset("ag_news")`. Expected validation
accuracy is **88–92%** at the same architecture and 5 epochs, training
time ~10 minutes on CPU. The bottleneck shifts from "can the model
solve this" to "does the model **generalize** beyond memorization", and
dropout / weight-decay become useful.

If staying with the synthetic dataset, the only meaningful follow-up
is to *make the templates harder*: overlap the topic-specific
vocabulary, introduce word-order ambiguity, and require multi-word
context. A worthwhile exercise but it has diminishing returns past
"the pipeline works".

## Reproducibility

From the repo root:

```bash
pip install -r requirements.txt
python src/train_text.py --config configs/lstm_text.yaml
```

Expected end state:

- `experiments/lstm_text/checkpoints/best.pt` — best-by-val checkpoint
  (~450 KB, 111,748 float32 parameters + vocab metadata).
- `experiments/lstm_text/logs/history.json` — per-epoch train/val loss
  and accuracy.

With `seed: 42` pinned, the same machine reproduces 100.00% val
accuracy from epoch 1 onward, with train loss decreasing from 0.39 →
0.0001 over 5 epochs.
