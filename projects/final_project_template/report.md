# Final Project — <Project Title>

> 2–4 pages. Match this skeleton; trim where you can. Anything you do
> not need, delete.

## Problem

One paragraph. What task, what dataset, what metric. Example:

> Multi-class classification of … on the … dataset (N classes,
> NN k train / NN k test). Metric: top-1 accuracy on the official test
> split.

## Method

One paragraph. The architecture in one sentence, then the recipe.
Example:

> A 6-layer CNN with `Conv → BN → ReLU → MaxPool` blocks and an
> AdaptiveAvgPool + Linear head. NN k parameters. Trained with
> AdamW(`lr=1e-3, weight_decay=5e-4`) for N epochs, batch size B,
> seed 42. Augmentation: RandomCrop + HorizontalFlip on the training
> set only.

Reference the canonical sources: `configs/main.yaml`, `src/models.py`
(your custom one if any), `src/train.py` (shared, repo-root).

## Results

### Headline metric

| Split  | Loss | Metric  |
|--------|-----:|--------:|
| Train  | …    | …       |
| **Val**| …    | **…**   |
| Test   | …    | …       |

Then a per-epoch table (or paste the `history.json` excerpt) so the
training curve can be reconstructed:

| Epoch | Train loss | Val loss | Val metric |
|------:|-----------:|---------:|-----------:|
| 1     | …          | …        | …          |
| 2     | …          | …        | …          |
| N     | …          | …        | …          |

### Training curve

Embed the saved figure:

```
![training curve](figures/training_curve.png)
```

Plot validation loss/metric vs. epoch with the best checkpoint marked.
If you cannot tell whether the model is under- or over-fitting from
this plot, re-read `docs/12_practical_methodology_debugging.md`.

### Reproducibility check

```
$ python src/evaluate.py --config configs/main.yaml --checkpoint experiments/<exp>/checkpoints/best.pt
...
re-evaluated val_loss=<X>  val_acc=<Y>
```

The number printed by `evaluate.py` must match the headline metric
above. If it does not, you saved the wrong checkpoint.

## Error analysis

- **Confusion matrix** (for classification): identify the one or two
  worst confused class pairs. Embed `figures/confusion_matrix.png`.
- **Per-class breakdown** (precision/recall): which class is the
  weakest, why.
- **A grid of failures**: 4×4 of misclassified examples + a sentence
  on the visible pattern.

If aggregate accuracy is your only number, the section is incomplete.

## What I would try next

One concrete experiment with a hypothesis. Example:

> Add OneCycleLR with `max_lr=1e-2` and `epochs=20`. Expected validation
> accuracy ≥ X% based on the literature for the same architecture and
> dataset family. If the gain is < 1 point, the bottleneck is the
> architecture, not the optimizer.

One item, not a wishlist.

## Reproducibility

From the repo root:

```bash
pip install -r requirements.txt

python src/train.py     --config projects/final_<your>/configs/main.yaml
python src/evaluate.py  --config projects/final_<your>/configs/main.yaml \
                        --checkpoint experiments/<exp>/checkpoints/best.pt
python src/inference.py --checkpoint experiments/<exp>/checkpoints/best.pt \
                        --input <file_or_dir>
```

Expected end state:

- `experiments/<exp>/checkpoints/best.pt` — best-by-val checkpoint
  (size and parameter count).
- `experiments/<exp>/logs/history.json` — per-epoch metrics that
  produced the curve above.

`seed: 42` (or whatever you pinned) reproduces the headline metric on
the same machine.
