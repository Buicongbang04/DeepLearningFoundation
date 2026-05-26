# Chapter 12 — Transformer (intro)

> The Transformer is *one* basic block — `Attention → FFN`, wrapped in residual + LayerNorm — stacked many times. Everything that looks complicated (BERT, GPT, ViT, T5) is the same block in a different arrangement.

## Mục tiêu (Goal)

After this chapter you can:

- Draw an encoder block and a decoder block from memory, labeling Attention, FFN, residual, and LayerNorm.
- Tell encoder-only, decoder-only, and encoder-decoder Transformers apart and name one model in each family.
- Explain why positional encoding is necessary and write a sinusoidal positional encoding by hand.
- Build a toy character-level language model with a small decoder-only Transformer.

## Why this chapter

This is the centerpiece. Almost every modern model — BERT, GPT-4, Claude, ViT, Whisper, Stable Diffusion's text encoder — is a Transformer of some shape. Once you can build the basic block, you can read any of these papers and understand the architecture in under an hour.

- **Builds on:** Chapter 11 (attention is the core sub-layer), Chapter 5 (LayerNorm, GELU), Chapter 7 (AdamW + warmup + cosine schedule, which Transformers basically require).
- **Sets up:** Chapter 13 (debugging deep models — Transformers are deep), and every downstream Computer Vision / NLP / LLM course.

## Key concepts

### Token embedding

A Transformer ingests integer token ids `(B, T)`, looks them up in an embedding table, and produces dense vectors `(B, T, d_model)`:

```python
self.embed = nn.Embedding(vocab_size, d_model)
x = self.embed(token_ids)              # (B, T) → (B, T, d_model)
```

`d_model` is the central dimension of the whole network: 512 for the original paper, 768 for BERT-base, 4096+ for modern LLMs. Every sub-layer is shaped so the input and output dimensions match `d_model`, which is what makes residual connections work everywhere.

### Positional encoding — why and how

Attention is **permutation-equivariant**: swap two tokens in the input and the output swaps the same way. The model has no built-in notion of "which token came first". For language, position is critical, so we *inject* positional information.

The original paper uses **sinusoidal positional encoding**:

```
PE(pos, 2i)   = sin( pos / 10000^{2i / d_model} )
PE(pos, 2i+1) = cos( pos / 10000^{2i / d_model} )
```

This produces a deterministic `(T, d_model)` matrix that is *added* to the token embeddings. Modern models (BERT, GPT-2) use *learned* positional embeddings instead — a separate `nn.Embedding(max_seq_len, d_model)` — which is simpler and works equally well in practice.

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):                              # x: (B, T, d_model)
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions)                  # broadcast over batch
```

(More modern variants — **RoPE**, **ALiBi** — are direction-only for this chapter.)

### The basic block (encoder version)

A Transformer encoder block has two sub-layers, each wrapped in a residual connection + LayerNorm:

```
y_1 = LayerNorm( x + MultiHeadSelfAttention(x) )       # sub-layer 1
y_2 = LayerNorm( y_1 + FFN(y_1) )                      # sub-layer 2
```

In code:

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.drop(a))
        x = self.ln2(x + self.drop(self.ffn(x)))
        return x
```

Pieces to internalize:

- **MultiHeadSelfAttention** — Chapter 11.
- **FFN** — two linear layers with a non-linearity in between. The hidden dimension `d_ff` is usually `4 × d_model`.
- **Residual** (`x + ...`) — every sub-layer adds its output to its input, preserving a gradient highway end-to-end.
- **LayerNorm** — normalizes across the feature dimension per token (not across the batch, unlike BatchNorm).

### Pre-norm vs. post-norm

The pattern above (`LN(x + sublayer(x))`) is **post-norm**, as in the original paper. Modern Transformers (GPT-2, ViT, etc.) use **pre-norm**:

```
y_1 = x + MultiHeadSelfAttention( LayerNorm(x) )
y_2 = y_1 + FFN( LayerNorm(y_1) )
```

Pre-norm trains much more stably at depth — you can train 24+ layers without warmup. Use pre-norm unless you have a specific reason not to.

### The decoder block

A decoder block is an encoder block with two changes:

1. The self-attention is **masked** (causal) — token `t` can only attend to tokens `≤ t`.
2. An extra **cross-attention** sub-layer attends to the encoder's output (only in encoder-decoder Transformers; pure decoder-only models like GPT do not have this).

```
y_1 = LN( x + MaskedSelfAttention(x) )
y_2 = LN( y_1 + CrossAttention(y_1, encoder_out) )   # only in encoder-decoder
y_3 = LN( y_2 + FFN(y_2) )
```

### Three flavors of Transformer

| Flavor             | Sub-layers per block             | Trained for                                      | Example          |
|--------------------|----------------------------------|--------------------------------------------------|------------------|
| Encoder-only       | self-attn (full) + FFN           | classification, embedding, masked-language model | BERT, RoBERTa, ViT |
| Decoder-only       | self-attn (causal) + FFN         | next-token prediction (language modeling)        | GPT-2 / 3 / 4, LLaMA, Claude |
| Encoder-decoder    | encoder blocks + decoder blocks with cross-attention | sequence-to-sequence (translation, summarization) | T5, BART, MarianMT |

Pick by *what you want to do*, not by what is fashionable:

- Classification / sentence embedding → encoder-only.
- Free-form text generation → decoder-only.
- Map one sequence to another with shared structure → encoder-decoder.

### Pretraining vs. fine-tuning

The dominant recipe across all flavors:

1. **Pretrain** on a huge unlabeled corpus with a self-supervised objective (masked-language modeling for BERT; next-token prediction for GPT; image-patch masking for MAE).
2. **Fine-tune** on a small labeled dataset for your downstream task. Either replace the head (like ResNet transfer learning in Ch 9) or use the model in-context with no parameter updates at all (LLM prompting).

For this course you will not pretrain a Transformer from scratch — you will load a pretrained checkpoint via `transformers` and fine-tune it.

### A toy decoder-only language model

The minimum viable Transformer for a character-level language model:

```python
class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4, max_len=256):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff=4 * d_model) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, idx):                          # idx: (B, T)
        B, T = idx.shape
        x = self.tok_embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        causal = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, attn_mask=causal)
        return self.head(self.ln_f(x))               # (B, T, vocab_size)
```

Training is just next-token cross-entropy:

```python
logits = model(idx[:, :-1])                          # predict idx[:, 1:] from idx[:, :-1]
loss = F.cross_entropy(logits.reshape(-1, vocab_size), idx[:, 1:].reshape(-1))
```

That is the *entire* core of a GPT-style model. Everything else (RMSNorm, RoPE, KV-cache, MoE) is engineering on top.

### Required training recipe

Transformers train very poorly with naive hyperparameters. The minimum recipe:

- Optimizer: **AdamW** with `weight_decay=0.1`, `betas=(0.9, 0.95)`.
- LR schedule: **linear warmup + cosine decay** (Ch 7).
- Gradient clipping: `max_norm=1.0`.
- Dropout: `0.1` in attention and FFN (drop during pretraining; can be reduced during fine-tuning).

Skip any of these and you will usually see "loss plateaus at random-baseline" failure modes.

## Common pitfalls

- Forgot positional encoding → attention is permutation-equivariant → model trains but cannot use word order → text quality is hopeless. Always add positions.
- Used post-norm at depth > 6–8 layers without warmup → loss diverges to NaN. Switch to pre-norm or add a long warmup.
- Causal mask wrong dimension or off-by-one → the decoder cheats by attending to the next token → train loss looks great, generation is gibberish. Verify by checking generation early in training.
- Mixed `(T, B, d)` with `(B, T, d)` between layers → silent shape bug; cross-entropy looks normal. Pick `batch_first=True` everywhere.
- `nn.MultiheadAttention` with `batch_first=False` (the default!) but input `(B, T, d)` → batch dim treated as time. Set `batch_first=True` explicitly.
- Forgot to scale loss by sequence length when comparing perplexities across runs → numbers not comparable. Always report `exp(mean_token_loss)`.

## Learning outcomes

- Sketch an encoder block (with residual + LayerNorm) and a decoder block (with causal mask + cross-attention) from memory.
- Identify encoder-only, decoder-only, and encoder-decoder by their training objective in one sentence each, and name BERT / GPT / T5 as canonical examples.
- Implement a tiny decoder-only Transformer for a character-level language model and observe the loss going below random baseline (`log(vocab_size)`).
- Diagnose a "loss diverges to NaN" Transformer training run by checking warmup, gradient clipping, and normalization placement.

## Quick check (self-test)

<details>
<summary>Q1 — Why does a Transformer need positional encoding at all?</summary>

Self-attention is permutation-equivariant — shuffling the input tokens shuffles the output the same way. Without an explicit position signal, the model has no way to distinguish "dog bites man" from "man bites dog".
</details>

<details>
<summary>Q2 — Name one model in each of the three flavors and what they are trained for.</summary>

- Encoder-only: BERT, trained with masked-language modeling, used for classification and embedding.
- Decoder-only: GPT, trained with next-token prediction, used for free-form generation.
- Encoder-decoder: T5, trained with span corruption / seq2seq, used for tasks like translation and summarization.
</details>

<details>
<summary>Q3 — Your Transformer's loss diverges to NaN after a few hundred steps. Two things to try first?</summary>

Add a linear warmup of a few hundred steps (LR ramps up from a near-zero value) and switch the block to *pre-norm*. Together these account for the vast majority of training-instability failures.
</details>

<details>
<summary>Q4 — In a decoder block, what is the difference between the masked self-attention and the cross-attention sub-layers?</summary>

Masked self-attention uses Q, K, V all from the *decoder* input and applies a causal mask. Cross-attention uses Q from the decoder and K, V from the *encoder*'s output, with no causal mask — the decoder can look at any source position.
</details>

## Further reading

- Vaswani et al., "Attention Is All You Need" (2017) — the original paper. Sections 3.2–3.4 are the core.
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018) — encoder-only.
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019) — decoder-only.
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020) — Transformer applied to vision.
- Karpathy, *nanoGPT* (GitHub) — minimal, well-commented decoder-only Transformer in PyTorch.

## Companion artifact

`projects/project_05_transformer_toy_language_model/` — a character-level decoder-only Transformer trained on a small text corpus, with generation samples.
