# Chapter 5 — Activation, Initialization, Normalization

> *Three things together decide whether a deep network trains at all.* Pick the activation, pick the initialization that matches it, and add normalization where the signal goes through too many layers. The wrong combination silently produces zero gradients or `nan`.

## Mục tiêu (Goal)

After this chapter you can:

- Visually compare sigmoid, tanh, ReLU, Leaky ReLU, and GELU — value and derivative.
- Explain vanishing and exploding gradients with a 1-layer-at-a-time signal-flow argument.
- Pick Xavier (Glorot) initialization for tanh/sigmoid and He (Kaiming) initialization for ReLU.
- Insert BatchNorm or LayerNorm in the right places and read the difference in training stability.

## Why this chapter

In Chapter 4 you wrote a 1-hidden-layer MLP and it trained. As soon as you push past 5-10 hidden layers, naïve choices break — gradients vanish, gradients explode, training diverges or stalls. The fix is not "use Adam"; the fix is the right activation × init × normalization combo. This chapter is *that combo*.

- **Builds on:** Chapter 4 (training loop), Chapter 2-3 (backprop intuition).
- **Sets up:** Chapter 6 (regularization), Chapter 8-9 (CNN — ReLU + He init + BatchNorm everywhere), Chapter 12 (Transformer — GELU + LayerNorm).

## Key concepts

### Activations — five you must recognize

| Activation | Formula                                                                 | Range      | Used for                            |
|------------|-------------------------------------------------------------------------|------------|-------------------------------------|
| Sigmoid    | `σ(z) = 1 / (1 + e^{-z})`                                               | `(0, 1)`   | binary output; rarely hidden today  |
| Tanh       | `tanh(z) = (e^z − e^{−z}) / (e^z + e^{−z})`                              | `(−1, 1)`  | older MLPs, RNN hidden state        |
| ReLU       | `max(0, z)`                                                              | `[0, +∞)`  | default hidden activation since 2012|
| LeakyReLU  | `z if z > 0 else 0.01·z`                                                | `(−∞, +∞)` | when ReLU produces dead neurons     |
| GELU       | `z · Φ(z)`, approx. `0.5·z·(1 + tanh(√(2/π)·(z + 0.044715·z³)))`         | `(−∞, +∞)` | modern Transformers (BERT, GPT)     |

Derivatives matter more than values:

- Sigmoid derivative `σ(z)(1 − σ(z))` peaks at `0.25` and goes to `0` quickly. *Vanishing gradient*.
- Tanh derivative `1 − tanh²(z)` peaks at `1` and decays. Less vanishing than sigmoid.
- ReLU derivative is `1` for `z > 0` and `0` for `z ≤ 0`. Either passes the gradient or kills it (*dead ReLU* problem).
- LeakyReLU keeps `0.01` for `z ≤ 0` to keep some gradient flowing.
- GELU is smooth; widely used in Transformers.

### Vanishing and exploding gradients

Backpropagation multiplies gradients layer by layer. In a deep network you get a product of `L` terms:

```
∂L/∂W_1 = (∂L/∂z_L) · ∏_{l=2..L}  ∂z_l/∂z_{l-1}
```

Each `∂z_l/∂z_{l-1}` is roughly `W_l^T · diag(σ'(z_{l-1}))`. Two failure modes:

- If `W_l` is too small **or** `σ'` is small (sigmoid/tanh in saturation), the product collapses to ~0 — **vanishing gradient**. Early layers stop learning.
- If `W_l` is too large, the product blows up — **exploding gradient**. Loss returns `nan`, training diverges.

The fix is to pick a *layer-wise variance* that keeps activations and gradients at unit scale through the network. That is what initialization schemes do.

### Xavier (Glorot) initialization — for sigmoid / tanh

Glorot & Bengio (2010) derived an initialization that keeps the variance of activations roughly constant from layer to layer assuming linear (or symmetric saturating) activations:

```
W ~ Uniform( −√(6 / (n_in + n_out)),  +√(6 / (n_in + n_out)) )
W ~ Normal(  0,  √(2 / (n_in + n_out)) )
```

In PyTorch:

```python
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

Use Xavier with **tanh** or **sigmoid** hidden layers.

### He (Kaiming) initialization — for ReLU

He et al. (2015) showed that ReLU halves the variance (negative half is zeroed out), so the right initialization is bigger:

```
W ~ Normal(0, √(2 / n_in))
```

In PyTorch:

```python
nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
nn.init.kaiming_normal_(layer.weight,  nonlinearity="relu")
```

Use Kaiming with **ReLU** or **LeakyReLU** hidden layers.

The PyTorch default for `nn.Linear` is already Kaiming-uniform with `a=√5`, which is close enough for most cases. Override when you have a specific reason.

### Normalization layers

Normalization keeps the *distribution* of activations (mean, variance) stable across the network. It is the third pillar that lets you train ~100-layer networks.

**BatchNorm (BN)** normalizes each feature across the *batch*:

```
ŷ_i = (x_i − μ_batch) / √(σ²_batch + ε)
y_i = γ · ŷ_i + β
```

Two learnable parameters `γ, β` per feature. Different statistics during train (batch stats) vs. eval (running stats). Used in nearly every CNN.

```python
nn.BatchNorm1d(num_features)   # MLP / 1-D input
nn.BatchNorm2d(num_channels)   # CNN feature map (B, C, H, W)
```

**LayerNorm (LN)** normalizes across the *feature* dimension within each example:

```
ŷ = (x − μ_example) / √(σ²_example + ε)
y = γ · ŷ + β
```

Used in Transformers and most NLP models. Stable regardless of batch size — important for very small (or single-example) batches.

```python
nn.LayerNorm(normalized_shape)
```

Other variants you will see later:

- **GroupNorm** — between BN and LN, normalizes within channel groups.
- **InstanceNorm** — used in style-transfer GANs.
- **RMSNorm** — common in modern LLMs.

### Where to put normalization

The classic convention is *Linear → BN → Activation*:

```python
nn.Sequential(
    nn.Linear(in_dim, h_dim),
    nn.BatchNorm1d(h_dim),
    nn.ReLU(),
)
```

There is some debate over BN-before-vs-after-activation. For this course we follow the original "BN-before-activation" convention. Transformers use the opposite *pre-norm* pattern (LayerNorm at the input of each sub-block).

### A practical recipe

For a brand-new deep MLP or CNN:

1. **Activation:** ReLU in hidden layers, softmax (multi-class) or sigmoid (binary) in output.
2. **Initialization:** Kaiming-normal for every `nn.Linear` and `nn.Conv2d`. PyTorch default is close enough.
3. **Normalization:** BatchNorm after each linear/conv layer (before activation).
4. **Optimizer:** Adam with `lr=1e-3` (Chapter 7).
5. **Sanity check:** the loss should *drop* in the first few iterations on a tiny batch. If it does not, debug *before* training longer.

## Common pitfalls

- "ReLU + Sigmoid output, all default init, no normalization" trains fine for 2 layers and refuses to train at 8 layers → you forgot Kaiming + BN.
- Loss diverges with `nan` after a few steps → exploding gradient → reduce learning rate, add gradient clipping, double-check init.
- BatchNorm hurts a tiny batch (batch size 1-2) → use LayerNorm or GroupNorm instead.
- A pretrained CNN's BN running stats degrade after fine-tuning a few epochs → freeze BN (`bn.eval()` and `requires_grad_(False)` on `γ, β`) on the backbone.
- "Dead ReLU" — some neurons always output 0 → switch to LeakyReLU, lower learning rate, or use Kaiming init correctly.

## Learning outcomes

- Plot sigmoid, tanh, ReLU, LeakyReLU, GELU and their derivatives on the same figure.
- Train the same MLP with sigmoid vs. ReLU and show which one is more stable on a deeper network.
- Initialize a `nn.Linear` with Xavier vs. Kaiming and explain why one is for tanh and the other for ReLU.
- Insert BatchNorm into an MLP and demonstrate that it speeds up convergence and stabilizes the loss curve.

## Quick check (self-test)

<details>
<summary>Q1 — Why is ReLU paired with He (Kaiming) initialization and not Xavier?</summary>

ReLU sets half of the activations to zero in expectation, which halves the variance through the layer. He initialization compensates by scaling the variance up by a factor of 2 (`√(2/n_in)` instead of `√(1/n_in)`).
</details>

<details>
<summary>Q2 — During inference, does BatchNorm use batch statistics or running statistics?</summary>

Running statistics (the moving averages it accumulated during training). You activate this mode with `model.eval()`.
</details>

<details>
<summary>Q3 — When would you prefer LayerNorm over BatchNorm?</summary>

When the batch size is small or variable (single-example inference, online learning), or in sequence models / Transformers where you want each token to be normalized independently.
</details>

<details>
<summary>Q4 — Your 30-layer MLP outputs <code>nan</code> after 50 iterations. List two likely causes.</summary>

(1) Learning rate is too large — gradients explode. (2) Initialization is too aggressive — early-layer activations blow up. Reduce LR, switch to Kaiming-normal, and add BatchNorm.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 6.3 (hidden units), Chapter 8.4 (parameter initialization), Chapter 8.7.1 (BatchNorm).
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010).
- He et al., "Delving Deep into Rectifiers" (2015) — Kaiming initialization.
- Ioffe & Szegedy, "Batch Normalization" (2015).
- PyTorch docs — `torch.nn.init`, `torch.nn.BatchNorm2d`, `torch.nn.LayerNorm`.

## Companion artifact

`notebooks/chapter_04_activation_init_norm.ipynb` — activation curves, vanishing-gradient demo on a deep MLP, init comparison, BN-on/off ablation.
