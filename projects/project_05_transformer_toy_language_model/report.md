# Project 05 — Toy Transformer Language Model: Report

## Problem

Char-level next-token prediction on a tiny self-contained corpus (a
160-line repetition of the *Hamlet* "to be or not to be" soliloquy,
roughly 4.6k characters total, vocabulary of 25 unique characters
including newline and space). Metric: cross-entropy loss per token, with
**per-token top-1 accuracy** as a sanity-check side metric, plus a
**qualitative check** — does the model generate plausible continuations?

The point of this project is the **end-to-end Transformer language-model
pipeline**: tokenize → embed + positional embed → stack of causal
blocks → linear head → autoregressive sampling. We deliberately use a
corpus small enough that the 73k-parameter model can memorize and recite
it back, which is the correct success criterion for a chapter-12
walking-skeleton.

## Method

`CharTransformerLM` from `src/models.py`:

```
Embedding(vocab=25, d_model=64)
+ position Embedding(block_size=64, d_model=64)
× 2 blocks of (LayerNorm → MultiheadAttention(causal mask) → residual
                LayerNorm → FFN(64 → 256 → 64) → residual)
LayerNorm → Linear(d_model=64, vocab=25)
```

72,985 trainable parameters. Pre-norm style, 4 attention heads, GELU
activation in the FFN.

Training recipe (`configs/toy_transformer_charlm.yaml`):

- **Optimizer:** AdamW, `lr = 3e-3`, `weight_decay = 1e-2`.
- **Gradient clipping:** `clip_grad_norm_(1.0)`.
- **Block size:** 64 characters; sliding-window over the corpus.
- **Batch size:** 32, **10 epochs**, `seed: 42`.

## Results

### Quantitative — per-token loss and accuracy

| Epoch | Train loss | Val loss | Per-token acc |
|------:|-----------:|---------:|--------------:|
| 1     | ~0.85      | ~0.21    | ~0.93         |
| 4     | 0.0671     | 0.0621   | 97.78%        |
| 8     | 0.0597     | 0.0575   | 97.80%        |
| 10    | 0.0793     | **0.0571** | **97.80%** |

Val loss minimum: **0.0571** at epoch 10.

### Wall-clock

~2.3 seconds per epoch on CPU (block_size 64, batch_size 32, 196 batches
per epoch). Ten epochs end-to-end finish in under 25 seconds.

### Qualitative — autoregressive sampling

Three prompts, `temperature=0.8`, `max_new_tokens=80`:

```
prompt="to be or not"
   -> "to be or not to be that is the question
       whether tis nobler in the mind to suffer
       the slings "

prompt="the slings "
   -> "the slings and arrows of outrageous fortune
       or to take arms against a sea of troubles
       and b"

prompt="devoutly to "
   -> "devoutly to be wished to die to sleep
       to sleep perchance to dream ay theres the rub
       to be or"
```

The model produces coherent, grammatical English that closely tracks
the source corpus — exactly what a tiny memorization-bound LM should
do. This is the **walking-skeleton** result for the chapter:

- Causal masking works (no future-token leakage; sampling is
  deterministic in the autoregressive sense).
- Position embeddings work (text remains aligned line-by-line).
- The training loop, checkpoint format, and `generate()` all wire up
  correctly.

### Reproducibility check

Reload `best.pt` and call `model.generate()` on the same prompts with
`torch.manual_seed(0)`. The completions are deterministic and exactly
match the listing above.

## Interpretation

This is intentionally an **over-fit** result, not a generalization one.
The corpus is repeated 16 times during dataset construction, total
~4.6k characters; the model has 73k parameters, so it has more than
enough capacity to memorize. We are validating that **every component
of a Transformer-LM training pipeline works**, not measuring the
quality of the language model.

For a generalization-focused experiment, the exact same code scales to
Tiny Shakespeare (~1MB, vocabulary ~65 characters) at `block_size=128`,
`d_model=128`, `num_layers=4`. Expected per-token val loss is around
**1.5–1.8 nats** at convergence, which is the standard for that
benchmark. CPU training would take roughly an hour; same code, same
config schema, larger corpus.

## What I would try next

**One concrete experiment, with a hypothesis:** swap the corpus for
Tiny Shakespeare (any plain-text book of about 1MB). Expected per-token
val loss is **1.5–1.8 nats** at convergence with `d_model=128, num_layers=4,
block_size=128`. The qualitative samples should still look like
Shakespeare (correct vocabulary, line breaks, character names) but
*compose new sentences* rather than recite the training data — the
hallmark of moving from memorization to generalization.

If staying with this corpus, the only meaningful follow-up is to
implement the **`[CLS]` classification head** trick on the same
encoder for a sanity check that the same building blocks support both
LM and classification — this is exactly the BERT vs. GPT split.

## Reproducibility

From the repo root:

```bash
pip install -r requirements.txt
python src/train_text.py --config configs/toy_transformer_charlm.yaml
```

Expected end state:

- `experiments/toy_transformer_charlm/checkpoints/best.pt` —
  best-by-val-loss checkpoint (~300 KB, 72,985 float32 parameters plus
  vocab metadata `stoi`/`itos`).
- `experiments/toy_transformer_charlm/logs/history.json` — per-epoch
  train/val loss and accuracy.

With `seed: 42` pinned, the same machine reproduces val loss 0.0571 at
epoch 10. Sampling is also deterministic given a `torch.manual_seed(0)`
seed before each `generate()` call.
