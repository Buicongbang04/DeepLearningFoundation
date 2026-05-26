# Chapter 10 — Sequence Modeling with RNN, GRU, LSTM

> A CNN treats an image as a *set* of patches; a recurrent network treats a sentence as an *order-dependent* stream of tokens. Before attention took over, RNNs were the only practical way to make a neural network *remember*.

## Mục tiêu (Goal)

After this chapter you can:

- Define sequence data, time step, hidden state, and the four input/output topologies (many-to-one, one-to-many, many-to-many, sync many-to-many).
- Write the forward pass of a vanilla RNN, GRU, and LSTM cell from memory.
- Explain *why* a vanilla RNN suffers from vanishing gradients and how the LSTM gating fixes it.
- Build a small LSTM text classifier in PyTorch with an embedding layer, packed sequences, and gradient clipping.

## Why this chapter

This is the *bridge* chapter between feed-forward networks (Ch 2–9) and attention/Transformer (Ch 11–12). Even though most state-of-the-art models in 2026 are Transformers, you still need recurrent intuition for three reasons:

1. **History.** Almost every NLP paper from 2014–2017 is RNN-based.
2. **Edge cases.** Streaming inputs, real-time audio, and very long sequences are still RNN territory.
3. **The pain it caused.** Attention exists *because* of vanishing gradients in RNNs. Without feeling that pain, the Transformer's motivation is opaque.

- **Builds on:** Chapter 7 (optimization, gradient clipping), Chapter 6 (regularization).
- **Sets up:** Chapter 11 (attention as a fix for RNN's long-range problem), Chapter 12 (Transformer drops recurrence entirely).

## Key concepts

### Sequence data and time step

A sequence is an ordered list of inputs `x_1, x_2, …, x_T`. Each `x_t` is a feature vector (or a token id); `t` is the time step. Examples:

- Text — `x_t` is a token id; `T` is the sentence length.
- Audio — `x_t` is a frequency-band vector at frame `t`.
- Time-series sensor — `x_t` is a sensor reading at second `t`.

In PyTorch, a batch of sequences is usually shaped `(B, T, F)` (batch-first) or `(T, B, F)` (time-first). Pick one convention and stick with it; mixing them is a top-five source of bugs.

### The recurrent connection

A vanilla RNN computes a hidden state `h_t` from the previous hidden state and the current input:

```
h_t = tanh( W_x · x_t + W_h · h_{t-1} + b )
y_t = W_y · h_t  + b_y               # optional output at each step
```

The same `W_x`, `W_h` matrices are *shared across time*. This is the recurrent analogue of "parameter sharing" in CNNs — the network applies the same transformation at every step.

```python
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")

    def forward(self, x, h_prev):
        return self.cell(x, h_prev)
```

In practice you use `nn.RNN`, `nn.GRU`, or `nn.LSTM`, which loop the cell over time internally:

```python
rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True)
out, h_T = rnn(x)           # x: (B, T, 128) → out: (B, T, 256), h_T: (1, B, 256)
```

### Unrolling and BPTT

Training an RNN means unrolling it through time and computing gradients with the chain rule. This is **Backpropagation Through Time (BPTT)**: conceptually identical to backprop on a feed-forward network *of depth T*. A 100-token sentence is effectively a 100-layer network during the backward pass.

For very long sequences, **truncated BPTT** chops the sequence into chunks (say 100 steps) and detaches the hidden state between chunks, so the gradient does not propagate further than one chunk:

```python
h = h.detach()        # detach between chunks; cuts the gradient graph
```

### The four input/output topologies

| Topology              | Picture                  | Example                                      |
|-----------------------|--------------------------|----------------------------------------------|
| many-to-one           | `[x_1 … x_T] → y`        | sentiment classification                     |
| one-to-many           | `x → [y_1 … y_T]`        | image captioning (image → caption)            |
| many-to-many (sync)   | `[x_t] → [y_t]`          | POS tagging, frame-level audio labeling       |
| many-to-many (encoder-decoder) | `[x] → [y]` with two RNNs | machine translation, seq2seq           |

Most teaching examples in this course are *many-to-one* (text classification) — simplest and shows the recurrent idea clearly.

### Vanishing and exploding gradients in vanilla RNN

The hidden state at the final step depends on all prior `h_{t-1}` via repeated multiplication by `W_h`. The gradient `∂L/∂h_t` for an early `t` therefore involves a product of many `W_h^T · diag(tanh')` terms. If the eigenvalues of `W_h` are smaller than 1 the product shrinks to zero (**vanishing**); if larger, it explodes (**exploding**).

Symptoms in practice:

- Loss goes down at first then plateaus. The model is using only the *last few tokens* and ignoring earlier context — vanishing.
- Loss occasionally jumps to a huge value (NaN, inf) — exploding. Fix with **gradient clipping** (Ch 7):

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Vanishing gradients are *not* fixed by clipping; they require an architecture change. That is what LSTM and GRU give you.

### GRU — Gated Recurrent Unit

GRU adds two gates: an **update gate** `z_t` and a **reset gate** `r_t`. They control how much of the past hidden state is kept vs. overwritten:

```
z_t = σ( W_z · [h_{t-1}, x_t] )           # how much to update
r_t = σ( W_r · [h_{t-1}, x_t] )           # how much past to forget
h̃_t = tanh( W · [r_t * h_{t-1}, x_t] )    # candidate new state
h_t = (1 − z_t) * h_{t-1} + z_t * h̃_t     # interpolate
```

Key insight: when `z_t ≈ 0`, the hidden state passes through unchanged (`h_t = h_{t-1}`). Gradient can flow back through this path *without* being multiplied by a weight matrix — much less vanishing.

```python
gru = nn.GRU(input_size=128, hidden_size=256, batch_first=True)
```

### LSTM — Long Short-Term Memory

LSTM is the older (1997) and more general sibling. It maintains a **cell state** `c_t` *separately* from the hidden state `h_t`, with three gates:

```
f_t = σ( W_f · [h_{t-1}, x_t] )            # forget gate
i_t = σ( W_i · [h_{t-1}, x_t] )            # input gate
o_t = σ( W_o · [h_{t-1}, x_t] )            # output gate
c̃_t = tanh( W_c · [h_{t-1}, x_t] )         # candidate cell
c_t = f_t * c_{t-1} + i_t * c̃_t            # new cell state
h_t = o_t * tanh(c_t)                      # new hidden state
```

`c_t` is the "memory highway": when `f_t ≈ 1` and `i_t ≈ 0`, the cell state just passes through. Gradient flows along this highway with minimal attenuation.

```python
lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
out, (h_T, c_T) = lstm(x)
```

GRU vs. LSTM: GRU has fewer parameters (no separate cell state), trains a bit faster, and is roughly competitive on most tasks. LSTM is the safer default when you are unsure.

### A small LSTM text classifier

The standard pipeline for many-to-one text classification:

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc    = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids, lengths):
        x = self.embed(token_ids)                                   # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_T, _) = self.lstm(packed)                             # h_T: (1, B, H)
        return self.fc(h_T.squeeze(0))                              # (B, num_classes)
```

Three pieces to internalize:

1. `nn.Embedding` — a learnable lookup table that maps a token id to a dense vector. `padding_idx=0` zeros out the gradient for pad tokens.
2. `pack_padded_sequence` — tells the LSTM the *true* length of each sequence in the batch so it skips pad tokens. Mandatory for variable-length input.
3. `h_T` — the hidden state at the last *real* token, used as the sentence representation.

## Common pitfalls

- Loss not decreasing on a long sequence → vanishing gradient → switch from `nn.RNN` to `nn.LSTM` or `nn.GRU`.
- Loss jumps to NaN → exploding gradient → add `clip_grad_norm_(..., max_norm=1.0)` before `optimizer.step()`.
- Forgot `pack_padded_sequence` → the LSTM treats pad tokens as real input → hidden state is contaminated by zeros at the end. Always pack variable-length batches.
- Used `padding_idx` but did not pass it to `nn.Embedding` → the embedding for the pad token is updated by gradient → small but real correctness bug.
- Mixed up `(B, T, F)` and `(T, B, F)` → silent: the model runs but treats the *batch* dimension as time. Always pick `batch_first=True` and stick with it.
- Saved `h_T` but used `out[:, -1]` for the last step on a *padded* batch → you get the hidden state at the *padding* position, not the last real token. Either pack or index by `lengths - 1`.

## Learning outcomes

- Write the forward pass of an RNN, GRU, and LSTM cell from memory.
- Identify vanishing vs. exploding gradient from a loss curve.
- Build an LSTM text classifier with embedding, packed sequences, and gradient clipping.
- Compare a vanilla RNN against an LSTM on a long-sequence task and quantify the gap.

## Quick check (self-test)

<details>
<summary>Q1 — In one sentence, why does a vanilla RNN have a vanishing-gradient problem?</summary>

The gradient at an early time step is a product of many `W_h^T · diag(tanh')` factors; if the eigenvalues of `W_h` are below 1, this product shrinks to zero exponentially in `T`, so the model can effectively only "see" the last few tokens.
</details>

<details>
<summary>Q2 — What does the forget gate `f_t` in an LSTM control?</summary>

How much of the previous cell state `c_{t-1}` survives into `c_t`. When `f_t ≈ 1`, the cell state is preserved; when `f_t ≈ 0`, it is wiped. This gate is what lets the LSTM "remember" relevant information across many steps.
</details>

<details>
<summary>Q3 — Your loss jumps from 1.3 to 14.0 in one step on an RNN. First fix?</summary>

Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`. This addresses exploding gradients, which is what the spike indicates.
</details>

<details>
<summary>Q4 — Why do we use `pack_padded_sequence` instead of letting the LSTM scan over pad tokens?</summary>

So the LSTM stops updating its hidden state once it reaches the end of each sequence's real tokens. Without packing, pad tokens contaminate the final hidden state (and waste compute).
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 10 (sequence modeling).
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) — original LSTM paper.
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014) — GRU paper and seq2seq.
- Andrej Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks* (blog, 2015).
- PyTorch docs — `torch.nn.RNN`, `torch.nn.GRU`, `torch.nn.LSTM`, `torch.nn.utils.rnn.pack_padded_sequence`.

## Companion artifact

`notebooks/chapter_09_rnn_lstm.ipynb` — vanilla RNN vs. LSTM on a long-dependency synthetic task; `projects/project_04_text_classification_lstm/` — end-to-end LSTM text classifier on a real dataset.
