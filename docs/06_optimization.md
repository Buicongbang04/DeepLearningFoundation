# Chapter 7 — Optimization for Deep Learning

> Picking the right optimizer is *not* the most important hyperparameter — the *learning rate* is. But knowing what each optimizer actually does is the difference between debugging in five minutes and debugging for five days.

## Mục tiêu (Goal)

After this chapter you can:

- Tell SGD, Momentum, Nesterov, RMSProp, Adam, and AdamW apart in one sentence each.
- Pick a learning rate by running a quick LR-range sweep.
- Add a learning-rate scheduler (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`) to a training loop.
- Apply gradient clipping for RNN-style models.

## Why this chapter

In Chapter 4 you used `optim.Adam(lr=1e-3)` and it worked. That is fine for MNIST, but for harder problems — CIFAR-10, ImageNet, Transformers — the choice of optimizer + scheduler is the difference between converging in 50 epochs and never converging. This chapter is the optimizer reference card.

- **Builds on:** Chapter 4 (training loop), Chapter 5 (activation/init/normalization), Chapter 6 (regularization).
- **Sets up:** every project from Chapter 8 onward. Transformer training in Chapter 12 specifically needs warmup + cosine schedule.

## Key concepts

### SGD — the base case

```
θ ← θ − η · ∇L(θ)
```

`η` is the learning rate. With **mini-batch SGD**, the gradient is computed on a batch of size `B` (a noisy estimate of the true gradient over the whole dataset).

In PyTorch:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
```

Properties:

- *Pro:* simple, well-understood, often generalizes best when tuned well.
- *Con:* slow on ill-conditioned loss surfaces; needs careful LR.

### Momentum

Maintain a velocity that accumulates past gradients:

```
v ← β · v + ∇L(θ)
θ ← θ − η · v
```

Typically `β = 0.9`. Momentum "smooths" the trajectory across noisy mini-batch gradients and lets you pick a larger effective step.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

### Nesterov momentum

A small twist on momentum: take the gradient at the *anticipated* position, not the current one.

```
θ_lookahead = θ − η · β · v_{t-1}
v ← β · v + ∇L(θ_lookahead)
θ ← θ − η · v
```

Often a free win over plain momentum.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
```

### RMSProp

Adaptive per-parameter learning rate based on a moving average of squared gradients:

```
s ← α · s + (1 − α) · ∇L(θ)²
θ ← θ − (η / √(s + ε)) · ∇L(θ)
```

Used heavily for RNNs before Adam took over.

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
```

### Adam — momentum + RMSProp

Maintain both a first-moment (momentum) and a second-moment (RMSProp) estimate:

```
m ← β1 · m + (1 − β1) · ∇L(θ)
v ← β2 · v + (1 − β2) · ∇L(θ)²
m̂ = m / (1 − β1^t)        # bias correction
v̂ = v / (1 − β2^t)
θ ← θ − η · m̂ / (√v̂ + ε)
```

Default: `β1=0.9, β2=0.999, ε=1e-8, lr=1e-3`. Adam is the *default* optimizer in modern deep learning because it converges fast with little tuning.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### AdamW — Adam with decoupled weight decay

`Adam(weight_decay=…)` mixes the L2 penalty into the gradient *before* the adaptive scaling, which weakens its effect on frequently-updated parameters. `AdamW` applies weight decay as a *separate* shrinkage step:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

Use AdamW whenever you would have used Adam with non-zero weight decay (which is almost always). It is the standard choice in Transformers and most modern vision models.

### Choosing a learning rate

A useful heuristic: do an **LR-range test**. Start with a very small LR, increase exponentially each iteration, plot loss vs. LR. The "knee" right before the loss explodes is your maximum stable LR; pick something a factor of 2-3 below it.

Rough defaults that usually work:

- Adam / AdamW on most problems: `lr = 1e-3` to `3e-4`.
- AdamW for Transformer fine-tuning: `lr = 2e-5` to `5e-5`.
- SGD with momentum: `lr = 1e-2` to `1e-1`, depending on batch size.

A 10× LR change is usually visible immediately in the first 1-2 epochs.

### Learning-rate schedulers

Constant LR is rarely best — most problems benefit from *decreasing* the LR as training progresses.

**StepLR** — drop by a factor every `step_size` epochs:

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

**CosineAnnealingLR** — smoothly anneal LR from `lr_max` to `lr_min` over `T_max` epochs:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

**OneCycleLR** — warm up to a peak LR, then anneal down. Excellent default for image classification.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
)
```

Plug it into the loop:

```python
for epoch in range(num_epochs):
    for x, y in train_loader:
        ...
        loss.backward()
        optimizer.step()
        scheduler.step()    # for OneCycleLR — per-step
    # scheduler.step()       # for StepLR / Cosine — per-epoch
```

Read the docs for *each* scheduler to know whether to call `.step()` per-step or per-epoch.

### Warmup

For Transformers and very large models, the loss surface near the random initialization is unstable. Linearly ramp up the LR from `0` to the target over the first few hundred iterations:

```python
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=500)
main   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main], milestones=[500])
```

### Gradient clipping

For RNNs and sometimes Transformers, gradients occasionally blow up to very large values. Clip them:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Two common modes:

- `clip_grad_norm_(params, max_norm)` — rescales the *global* gradient norm to at most `max_norm`.
- `clip_grad_value_(params, clip_value)` — clips each component to `[-clip_value, +clip_value]`.

For LSTMs, `max_norm=1.0` to `5.0` is typical.

### An optimizer cheat sheet

| Optimizer  | LR default | When to use                                               |
|------------|------------|-----------------------------------------------------------|
| SGD + Mom  | 1e-2       | Image classification with big datasets (ImageNet ResNet). |
| Adam       | 1e-3       | Sane default for everything.                              |
| AdamW      | 1e-3       | Default for Transformers, modern vision models.           |
| RMSProp    | 1e-3       | Older RNN work; rarely chosen new today.                  |
| LBFGS      | n/a        | Small-batch fine-tuning where curvature info helps.       |

## Common pitfalls

- "Adam with `weight_decay`" → weight decay is undercounted; use `AdamW` instead.
- LR too large → loss diverges or oscillates → halve until stable.
- LR too small → loss decreases slowly → double until you find the largest stable rate.
- Forgot `scheduler.step()` → LR stays constant. Print `scheduler.get_last_lr()` once per epoch to confirm.
- Gradient clipping inside a `no_grad` block → no effect. Clip *before* `optimizer.step()` and *after* `loss.backward()`.
- Using `OneCycleLR` but stepping per-epoch instead of per-step → schedule never finishes → loss curve looks suspicious.

## Learning outcomes

- Train the same model with SGD, SGD+momentum, Adam, and AdamW and rank them by final validation accuracy.
- Run a 1-epoch LR-range test to pick a learning rate.
- Plug a `CosineAnnealingLR` or `OneCycleLR` into the training loop.
- Add `clip_grad_norm_(model.parameters(), 1.0)` to an LSTM training script and observe whether the loss is more stable.

## Quick check (self-test)

<details>
<summary>Q1 — Why is the learning rate usually more important than the choice of optimizer?</summary>

Within a reasonable optimizer family, a well-tuned LR almost always beats the wrong LR on a "better" optimizer. The LR controls the *size* of every update, which is the dominant effect on training dynamics.
</details>

<details>
<summary>Q2 — In one sentence, what is the difference between SGD+momentum and Adam?</summary>

SGD+momentum applies the same learning rate to every parameter; Adam applies a per-parameter adaptive learning rate based on the moving average of past gradient magnitudes.
</details>

<details>
<summary>Q3 — When training an LSTM, training loss occasionally jumps from 1.5 to 12.0 in a single step. What do you add?</summary>

Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`. This caps the global gradient norm and prevents these spikes.
</details>

<details>
<summary>Q4 — You are using <code>OneCycleLR</code> and your loss curve looks flat. Most likely bug?</summary>

You are calling `scheduler.step()` per-epoch instead of per-step. `OneCycleLR` is parametrized by `steps_per_epoch × epochs`, so it needs a `.step()` after every batch.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 8 (optimization).
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014).
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, 2019).
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) — the LR-range test.
- PyTorch docs — `torch.optim`, `torch.optim.lr_scheduler`.

## Companion artifact

`notebooks/chapter_06_optimization_adam_sgd_scheduler.ipynb` — head-to-head optimizer comparison on Fashion-MNIST, LR-range test plot, three-scheduler ablation.
