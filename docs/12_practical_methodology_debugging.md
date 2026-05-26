# Chapter 13 — Practical Methodology: Debugging Deep Models

> A model that is "not learning" is almost never a deep, novel research problem. It is almost always a data bug, a shape bug, a learning-rate bug, or a metric bug — in that order of frequency. The skill is to *check the boring things first*.

## Mục tiêu (Goal)

After this chapter you can:

- Run a 20-item debug checklist on a model that is not learning.
- Tell a *data bug*, a *model bug*, an *optimizer bug*, and a *metric bug* apart from their symptoms.
- Use the "overfit a tiny batch" sanity check to localize a failure in under five minutes.
- Log gradient norm, weight norm, and a few key metrics during training so the next failure is easier to diagnose.

## Why this chapter

Up to now you have learned components: tensors, MLPs, CNNs, RNNs, attention, Transformers. This chapter is about the *workflow* that connects them. The biggest skill gap between a beginner and an intermediate practitioner is not architectural — it is the ability to look at a flat loss curve and know *which knob to turn first*.

- **Builds on:** every previous chapter — debugging requires you to know what *should* happen so you can spot what *is* happening.
- **Sets up:** Chapter 16 (deployment) and Chapter 17 (final project), which both demand a model that actually works.

## Key concepts

### The four bug families

When a model is not learning, the failure is almost always in one of four places:

| Family       | Symptom                                         | Where to look                                    |
|--------------|-------------------------------------------------|--------------------------------------------------|
| Data bug     | Loss flat at random baseline from epoch 1       | DataLoader output, label mapping, normalization. |
| Model bug    | Loss decreases on train but never on val        | Architecture, regularization, capacity.          |
| Optimizer bug| Loss diverges, oscillates, or plateaus mid-run  | Learning rate, scheduler, gradient norm.         |
| Metric bug   | Loss looks fine but metric is suspicious        | Argmax direction, class index mapping, reduction.|

The next sections give one diagnostic for each.

### Sanity check #1 — overfit a tiny batch

This single check finds 80% of all bugs. Take **2–8 examples** from your training set, *no shuffling, no augmentation, no dropout, no validation*, and train for 200–500 steps. The model should reach near-zero loss.

```python
tiny_x, tiny_y = next(iter(train_loader))     # one batch
tiny_x, tiny_y = tiny_x[:8], tiny_y[:8]

for step in range(500):
    optimizer.zero_grad()
    loss = criterion(model(tiny_x), tiny_y)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(step, loss.item())
```

Interpretations:

- Loss goes to ~0 → architecture and optimizer are fine; the bug is in *the rest of the data* (more data, augmentation, regularization, train/val mismatch).
- Loss plateaus at a non-trivial value → the model cannot represent the function. Increase capacity, check activation/output shape, inspect the data.
- Loss is NaN / inf → exploding gradient or bad initialization → check `lr`, add clipping, reduce LR.
- Loss is random and flat → the labels are not what the model thinks they are, or shapes are wrong.

### Sanity check #2 — print shapes everywhere

Most "model bug" reports are actually silent shape bugs that produced a runtime success but garbage gradients. Add shape prints at every layer boundary the first time you run a new architecture:

```python
def forward(self, x):
    print("input:", x.shape)
    x = self.conv(x);  print("after conv:", x.shape)
    x = self.pool(x);  print("after pool:", x.shape)
    x = x.flatten(1);  print("after flatten:", x.shape)
    return self.head(x)
```

Remove or wrap with `if self.debug:` once stable. The 30 seconds you spend printing now saves an hour later.

### Sanity check #3 — random-baseline loss

You should always know the *expected loss for a random classifier* and compare:

```
Random baseline loss = log(num_classes)        # for softmax + cross-entropy
```

For 10-class MNIST that is `log 10 ≈ 2.30`. If after one epoch your loss is `2.30 ± 0.05`, the model is not learning at all — it is outputting a uniform distribution. If it is *above* the random baseline, your label encoding is probably broken.

### Train/val curves — reading the four shapes

| Pattern                                 | Diagnosis                                         | Next move                                       |
|-----------------------------------------|---------------------------------------------------|-------------------------------------------------|
| Train ↓, val ↓ (both converging)        | Healthy. Pick the best-val checkpoint.            | Stop early, ship.                               |
| Train ↓, val flat or ↑                  | Overfitting.                                       | More data, augmentation, dropout, weight decay. |
| Train flat, val flat                    | Underfitting / not learning.                       | Increase capacity, check LR, run tiny-batch.    |
| Train ↓, val ↓, then val jumps up       | LR schedule mis-set, BN running stats unstable, or a bad data sample in val. | Inspect at the spike step.                      |

Plot both curves on the *same* axes — that comparison is the single most informative chart in deep learning.

### Logging gradient norm

Most optimizer bugs are visible as anomalies in the gradient norm. Log it every step:

```python
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.detach().pow(2).sum().item()
total_norm = total_norm ** 0.5
```

What to look for:

- Norm grows monotonically by orders of magnitude → about to explode → clip or lower LR.
- Norm collapses to near zero in the first few epochs → vanishing → check activations, switch optimizer, raise LR.
- Norm is wildly different across layers → some layers are dead. Use Xavier/He initialization and BatchNorm/LayerNorm appropriately (Ch 5).

For per-layer breakdown, log `p.grad.norm()` for each named parameter and plot a histogram.

### Data leakage — the silent killer

A model with 99% test accuracy that bombs in production almost always has a data-leakage bug. Sources to check:

- **Train/val split done after augmentation.** Same source image ends up in both train and val under different crops.
- **Label leakage in features.** A "patient ID" feature was created from the target; a "timestamp" feature lets you trivially identify the class.
- **Normalization computed on the full dataset.** Mean and std should be computed on the *training set only*, then applied to val/test.
- **Time-series shuffle.** Shuffling time-series data leaks future information into the training set.

Heuristic: if val accuracy is *implausibly high* (>95% on a task that the literature reports at 80%), suspect leakage before celebrating.

### `model.train()` / `model.eval()` and `no_grad`

Two state switches that, when forgotten, silently break inference:

- `model.train()` — turns dropout *on* and uses batch statistics in BatchNorm. Set at the top of every training epoch.
- `model.eval()` — turns dropout *off* and uses BN's running statistics. Set before every validation/test/inference pass.
- `torch.no_grad()` — disables autograd graph construction. Use during evaluation/inference to save memory and time.

```python
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        ...
```

Forgetting `model.eval()` is the cause of "my val accuracy is 30 points lower than my train accuracy on the same data" — dropout is still firing during eval.

### Experiment tracking

Log every run to a central place so you can compare. Minimum to log:

- The config (LR, batch size, optimizer, scheduler, seed).
- Train loss, val loss, val metric per epoch.
- Gradient norm and weight norm per epoch.
- The git commit hash of the code that produced the run.
- Hardware (CUDA device name) — for reproducibility across machines.

Two viable options:

- **TensorBoard** — `SummaryWriter` shipped with PyTorch. Simple, local files.
- **MLflow** / **Weights & Biases** — cloud-backed; richer UI; team-friendly.

Either is fine. The mistake is using *neither* and trying to remember 30 runs from terminal logs.

### The 20-item debug checklist

Run this list, top to bottom, when a model is not learning. Stop as soon as you find the bug.

1. Is the loss above `log(num_classes)`? If yes, label encoding is broken.
2. Can you overfit 4 examples in 500 steps? If no, the architecture or LR is broken.
3. Print one batch from the DataLoader. Do labels match images?
4. Is `dtype` of inputs `float32`? Of labels (for classification) `int64`?
5. Is the input on the same device as the model? `next(model.parameters()).device`
6. Is `model.train()` set during training and `model.eval()` during validation?
7. Did you call `optimizer.zero_grad()` at the start of every step?
8. Did you call `loss.backward()` and `optimizer.step()` in that order?
9. Is the LR sane? Try `1e-3` for Adam, `1e-2` for SGD, sweep one decade up and down.
10. Are gradients flowing? Log `total_grad_norm`; it should be positive and stable.
11. Did you normalize inputs the *same way* in train and val?
12. Did you compute val normalization statistics on the *training set only*?
13. Is the loss function appropriate? Cross-entropy expects raw logits, not softmaxed probabilities.
14. Is the metric direction right? (Accuracy uses argmax over class dim; reductions match the loss.)
15. Did you set `random.seed`, `numpy.random.seed`, and `torch.manual_seed` for reproducibility?
16. Is dropout off during eval? Is BatchNorm using running stats?
17. Does the train/val split leak (same image, augmented twice; time-series shuffle)?
18. Is the model capacity sane? Print `sum(p.numel() for p in model.parameters())`.
19. Does the val loss diverge from the train loss after epoch N? You are overfitting; add regularization.
20. Have you saved and reloaded a checkpoint, then re-evaluated? Do you get the same metric?

Memorize this list. It will save you days over a career.

## Common pitfalls

- "I tried a bigger model and it did not help" → almost certainly a data or shape bug. Bigger models hide bugs, they do not fix them.
- "My train accuracy is 99% but val is 30%" → either overfitting (Ch 6) *or* `model.eval()` is not being called (dropout still on). Check both.
- "My val loss is *worse* than random" → label encoding has an off-by-one, or argmax is along the wrong dimension. Inspect one prediction by hand.
- "I changed the LR and it did not help" → did you change it by *one decade* (10×), or by 20%? Less than 2× rarely makes a visible difference.
- "Loss looks fine, accuracy is 0%" → wrong reduction in the metric (per-batch vs per-epoch), wrong class index, or one-hot labels passed to a sparse loss. Print one prediction next to its label.

## Learning outcomes

- Recite the 20-item debug checklist in roughly the right order.
- Sort a failure into data/model/optimizer/metric bug by symptom alone.
- Run the "overfit a tiny batch" sanity check and interpret the result.
- Log gradient norm in your training loop and read it as a diagnostic.

## Quick check (self-test)

<details>
<summary>Q1 — Your loss after one epoch is exactly `log(num_classes)`. What is happening?</summary>

The model is outputting a uniform distribution over classes — it has not learned anything. Either the data is broken (random labels, wrong mapping), the LR is too small to make any update, or gradients are not flowing. Run the tiny-batch overfit test next.
</details>

<details>
<summary>Q2 — Train accuracy is 99%, val accuracy is 30%. What are the two most likely causes?</summary>

Either overfitting (true generalization gap — fix with augmentation, dropout, weight decay, more data), or you forgot `model.eval()` during validation so dropout is still firing on val. Check the `.eval()` call first because it is a one-line fix.
</details>

<details>
<summary>Q3 — You cannot overfit a tiny batch in 500 steps. What does that tell you?</summary>

The bug is *not* about generalization — it is in the architecture, the loss, or the optimizer. Either the model cannot represent the function, the loss is wired wrong (e.g. softmax-after-CE), or the LR is too small to move parameters at all.
</details>

<details>
<summary>Q4 — Your gradient norm grows from 1.0 to 10⁴ over the first 50 steps. What do you do?</summary>

Reduce the learning rate by 10× and/or add `clip_grad_norm_(..., max_norm=1.0)`. The growing norm is the precursor to a NaN explosion; clip *and* lower LR is more robust than either alone.
</details>

<details>
<summary>Q5 — You get 99% val accuracy on a task the literature reports at 80%. What is the first thing to check?</summary>

Data leakage. The most common form is that the val split shares augmented copies of train images, or a feature derived from the target leaks into the inputs. Implausibly good numbers are almost always a leakage bug.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 11 (practical methodology).
- Andrej Karpathy, *A Recipe for Training Neural Networks* (blog, 2019) — the seminal write-up of this kind of methodology.
- "Things that confused me" lists in your own lab notebook — keep one.
- PyTorch docs — `torch.nn.utils.clip_grad_norm_`, `torch.utils.tensorboard`.

## Companion artifact

`labs/lab_06_experiment_tracking.ipynb` — wire TensorBoard into the standard training loop, log gradient norm and val metric, sweep a few learning rates and compare runs.
