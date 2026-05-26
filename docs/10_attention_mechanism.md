# Chapter 11 — Attention Mechanism

> Attention is **content-addressable memory**: for each query position, look at every other position, decide how relevant it is, and pull a weighted average. The Transformer is just this idea, repeated, stacked, and scaled.

## Mục tiêu (Goal)

After this chapter you can:

- State the Query / Key / Value idea in one sentence and explain what each is a projection *of*.
- Implement scaled dot-product attention in PyTorch from scratch (no `nn.MultiheadAttention`).
- Distinguish self-attention from cross-attention, and explain why multi-head attention helps.
- Plot an attention matrix on a toy sequence and read which tokens attended to which.

## Why this chapter

Chapter 10 showed that RNNs struggle with long-range dependencies even with LSTM gating: information from token 1 has to thread through 100 sequential cells to reach token 100. **Attention** removes the bottleneck by letting every token directly access every other token, in one step, in parallel.

This chapter introduces attention as an *isolated mechanism* — no positional encoding, no LayerNorm, no FFN. Chapter 12 puts it all together into the Transformer.

- **Builds on:** Chapter 10 (the long-range problem in RNNs), Chapter 5 (softmax, normalization), Chapter 4 (broadcasting and matrix multiplication shapes).
- **Sets up:** Chapter 12 (attention is the core of every Transformer block).

## Key concepts

### The long-range problem

In an LSTM, the influence of token 1 on the output at token 100 must pass through 99 sequential cells. Each cell may attenuate the signal slightly, and the path is *fundamentally serial* — you cannot parallelize across the time axis during the forward pass.

Attention proposes a different shape of computation: at position `t`, directly look at *all* positions and form a weighted average. The distance from token 1 to token 100 is now one operation, not 99.

### Query, Key, Value — the three roles

Think of attention as a soft dictionary lookup:

- A **Query** asks "what am I looking for?"
- A **Key** advertises "this is what I am."
- A **Value** delivers "this is what I carry."

For each query, compare it against every key to get a similarity score, softmax those scores into weights, then take a weighted sum of the corresponding values. The same input tensor `X` is *projected* by three separate learnable matrices into Q, K, V:

```
Q = X · W_Q
K = X · W_K
V = X · W_V
```

This is why the chapter title says Q, K, V are *projections of* the input — they are three different views of the same tokens.

### Dot-product attention

Given a single query `q ∈ R^d` and a set of keys `K ∈ R^{T × d}` and values `V ∈ R^{T × d_v}`:

```
scores = q · K^T            # shape: (T,)
weights = softmax(scores)   # shape: (T,), sums to 1
out = weights · V           # shape: (d_v,)
```

The output is a *convex combination* of the values, weighted by how relevant each key is to the query. With a batch of `T_q` queries `Q ∈ R^{T_q × d}`:

```
scores = Q · K^T            # shape: (T_q, T_k)
weights = softmax(scores, dim=-1)
out = weights · V           # shape: (T_q, d_v)
```

### Scaled dot-product attention

For high `d`, the dot product `q · k` has variance that grows linearly with `d`. The softmax of a large-variance input becomes nearly one-hot, which kills the gradient. The fix is a constant rescale:

```
Attention(Q, K, V) = softmax( (Q · K^T) / √d_k ) · V
```

The `/ √d_k` term keeps `Var(q · k)` close to 1 regardless of dimension. Without it, training a Transformer with `d_k > 64` becomes flaky.

In PyTorch, from scratch:

```python
import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K: (B, T, d_k)   V: (B, T, d_v)
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)     # (B, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)                   # (B, T, T)
    return weights @ V, weights                           # ((B, T, d_v), (B, T, T))
```

Two implementation details worth knowing:

- The `mask == 0` positions get `−∞` *before* softmax so they receive weight 0 *after* softmax.
- Returning `weights` is optional but extremely useful for visualization — that is the attention matrix you plot.

### Self-attention vs. cross-attention

- **Self-attention** — Q, K, V all come from the *same* sequence `X`. Used in the encoder, and in the masked-self-attention sub-layer of the decoder.
- **Cross-attention** — Q comes from the *target* sequence; K and V come from the *source* sequence. Used in the encoder-decoder layer of seq2seq Transformers (the decoder asks the encoder "what should I attend to in the input?").

In both cases the math is the same; only the input sequences differ.

### Causal (masked) attention

For language modeling, position `t` is not allowed to look at positions `> t` (otherwise the model cheats). Apply a lower-triangular mask before softmax:

```python
T = scores.size(-1)
causal_mask = torch.tril(torch.ones(T, T, device=scores.device)).bool()
scores = scores.masked_fill(~causal_mask, float("-inf"))
```

This turns the attention matrix into a lower-triangular shape: each row only sees itself and earlier columns.

### Multi-head attention (intuition)

A single attention head computes one set of weights. **Multi-head attention** runs `h` heads in parallel, each with its own `W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}` matrices projecting into a smaller dimension `d_k = d_model / h`:

```
head_i = Attention(X · W_Q^{(i)}, X · W_K^{(i)}, X · W_V^{(i)})
MultiHead(X) = Concat(head_1, …, head_h) · W_O
```

Why multiple heads? Different heads can specialize in different kinds of relationships — one head tracks subject-verb agreement, another tracks coreference, another tracks position. Concatenating gives the next layer a richer representation than a single head could provide.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]                 # each: (B, h, T, d_k)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).reshape(B, T, -1)      # (B, T, d_model)
        return self.out(out)
```

Reading the shapes carefully is more useful here than reading the words.

### Attention matrices as a debug tool

The `weights` matrix `(T, T)` is one of the most informative debugging artifacts in deep learning. Plot it as a heatmap:

```python
import matplotlib.pyplot as plt
plt.imshow(weights[0, 0].detach().cpu(), aspect="auto")
plt.colorbar()
plt.xlabel("Key position")
plt.ylabel("Query position")
plt.show()
```

What to look for:

- A diagonal band — the head is mostly attending to neighbors.
- Vertical stripes — many queries attend to one specific key (a "sink" token like `[CLS]` or `[SEP]`).
- Block-diagonal structure — heads have segmented the sequence into chunks.
- Uniform gray — the head has degenerated and is attending uniformly (often a sign of dead heads).

## Common pitfalls

- Forgot the `/ √d_k` scaling → softmax saturates → gradients vanish → loss decreases very slowly. Always scale.
- Used `mask == 0` to fill with `-inf` but the mask is `True`/`False` typed → comparison silently wrong. Use `~mask` or `mask == False` consistently.
- Mask broadcasted wrong (`(T, T)` vs `(B, h, T, T)`) → silent NaNs from `-inf` propagation. Print mask shape next to scores shape.
- Causal mask off-by-one — using `torch.tril(...)` strictly-lower instead of lower-triangular blocks token `t` from attending to itself. Use `torch.tril(..., diagonal=0)`.
- Permuted Q, K, V incorrectly in multi-head → effective `T` and `h` get swapped → runtime succeeds, accuracy is garbage. Always check shape `(B, h, T, d_k)` before the matmul.
- Reading the attention matrix as "model attention" in a literal sense → attention weights correlate with importance but are not a faithful explanation. Treat them as a debug aid, not ground truth.

## Learning outcomes

- Explain Q, K, V as three learnable projections of the same input.
- Implement scaled dot-product attention in 5–10 lines of PyTorch.
- Apply a causal mask correctly for language modeling.
- Plot and interpret an attention matrix on a toy sequence.

## Quick check (self-test)

<details>
<summary>Q1 — In one sentence, what does the softmax in scaled dot-product attention compute?</summary>

A probability distribution over the keys for each query — these are the *attention weights*, summing to 1 over the key dimension, that say how much each key (and its value) contributes to the output for that query.
</details>

<details>
<summary>Q2 — Why do we divide by `√d_k` before the softmax?</summary>

Without it, the variance of the dot product `q · k` grows linearly with `d_k`, pushing the softmax into a saturated region where its gradient is near zero. Scaling by `√d_k` keeps the variance roughly constant and the gradient healthy.
</details>

<details>
<summary>Q3 — Why use multiple heads instead of one head with a larger `d_k`?</summary>

Because attention is *linear* given the weights — averaging in one big subspace cannot represent multiple distinct relationships. Multiple heads project into different subspaces so each can attend along a different axis (e.g., syntactic vs. semantic), and the concatenation provides a richer next-layer input.
</details>

<details>
<summary>Q4 — What does a causal mask look like on the attention matrix?</summary>

A lower-triangular matrix (including the diagonal) of `1`s with `0` above the diagonal. Before softmax, the `0` positions become `-inf`; after softmax they are exactly 0, so token `t` only attends to tokens `≤ t`.
</details>

## Further reading

- Bahdanau, Cho, Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014) — the original attention paper for seq2seq.
- Vaswani et al., "Attention Is All You Need" (2017) — Section 3.2 defines scaled dot-product and multi-head attention.
- Jay Alammar, *The Illustrated Transformer* (blog) — the most-cited visual explanation.
- Karpathy, *Let's build GPT: from scratch, in code, spelled out* (YouTube) — implements scaled dot-product attention live.

## Companion artifact

`notebooks/chapter_10_attention_toy.ipynb` — implement scaled dot-product attention, run it on a toy sequence, plot the attention heatmap.
