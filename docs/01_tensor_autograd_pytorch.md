# Chapter 1 — PyTorch Warmup: Tensor, Device, Autograd

> Everything in PyTorch is a tensor. Everything trainable in PyTorch needs `requires_grad`. Everything fast needs `.to(device)`. Get these three reflexes into your fingers before touching neural networks.

## Mục tiêu (Goal)

After this chapter you can:

- Create 1-D, 2-D, and 4-D tensors and read off their shape, dtype, and device.
- Move a tensor between CPU and GPU using `.to(device)`.
- Compute a gradient with `autograd` and cross-check it against a derivative you take by hand.
- Explain when `requires_grad=True`, `.backward()`, `.grad`, and `torch.no_grad()` are needed.

## Why this chapter

PyTorch is the language we will speak for the rest of the course. There are three primitives you cannot skip:

1. The tensor — *the* data structure (every input, parameter, gradient, and output is a tensor).
2. The device — CPU vs. GPU, which decides how fast (or whether) anything runs.
3. The computational graph — built dynamically as you call ops, and consumed by `.backward()` to compute gradients.

If those three are clear, the rest of the course is engineering. If they are not, every later bug will look like magic.

- **Builds on:** NumPy (shape, broadcasting, slicing).
- **Sets up:** Chapter 2 (MLP), Chapter 3 (manual + autograd backprop), Chapter 4 (training loop).

## Key concepts

### The tensor — shape, dtype, device

A tensor is a multi-dimensional array, like a NumPy array, plus two extras: a `device` and an optional `grad`.

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(x.shape)   # torch.Size([2, 3])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

Three properties to read off every tensor you ever see:

- `shape` — the dimension sizes. `(2, 3)` means 2 rows, 3 columns. Order matters.
- `dtype` — `float32`, `float64`, `int64`, `bool`. Always check before passing into a loss function (`CrossEntropyLoss` wants integer labels, not floats).
- `device` — `cpu` or `cuda:0`. Mismatched devices throw `RuntimeError: Expected all tensors to be on the same device`.

### Common ways to create a tensor

```python
torch.zeros(2, 3)              # all zeros
torch.ones(2, 3)               # all ones
torch.eye(3)                   # identity matrix
torch.arange(0, 10, 2)         # 0, 2, 4, 6, 8
torch.linspace(0, 1, 5)        # 5 evenly-spaced points in [0, 1]
torch.randn(2, 3)              # standard normal
torch.randint(0, 10, (2, 3))   # ints in [0, 10)

# From a Python list or a NumPy array.
torch.tensor([[1, 2], [3, 4]])
torch.from_numpy(np_array)
```

### Tensor shapes you will see again and again

| Shape         | Meaning                                                  | Where it shows up                   |
|---------------|----------------------------------------------------------|-------------------------------------|
| `(B,)`        | batch of scalars                                          | regression target, class labels     |
| `(B, D)`      | batch of D-dim vectors                                   | MLP input/output, embeddings        |
| `(B, C, H, W)`| batch of images                                           | CNN input/output                    |
| `(B, T, D)`   | batch of sequences of length `T` with feature dim `D`     | RNN, Transformer                    |

`B` = batch, `C` = channels, `H/W` = height/width, `T` = time step, `D` = feature.

### Reshaping — `view`, `reshape`, `permute`, `squeeze`, `unsqueeze`

```python
x = torch.randn(2, 3, 4)

x.view(2, 12)        # (2, 12) — same data, no copy. Requires contiguous memory.
x.reshape(2, 12)     # same shape, but works even if x is non-contiguous (may copy).
x.permute(0, 2, 1)   # swap dim 1 and dim 2 → (2, 4, 3)
x.squeeze(0)         # drop a singleton dim, e.g. (1, 3, 4) → (3, 4)
x.unsqueeze(0)       # add a singleton dim at position 0 → (1, 2, 3, 4)
x.flatten(start_dim=1)  # collapse all dims from 1 onwards → (2, 12)
```

Rule of thumb: use `view` when you know the tensor is contiguous (it is, right after `randn`), otherwise use `reshape`.

### CPU vs. GPU — `.to(device)`

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(2, 3).to(device)
model = MyModel().to(device)
```

Three rules:

1. Decide `device` *once* at the top of the script.
2. Move *both* model and data to `device` *before* the training loop.
3. Bring tensors back with `.cpu()` only when you need them for NumPy, plotting, or saving.

### Broadcasting

PyTorch follows NumPy broadcasting rules. Two shapes are compatible if, starting from the right, every pair is either equal, or one of them is `1`.

```python
a = torch.randn(3, 1)
b = torch.randn(1, 4)
c = a + b              # broadcasts to (3, 4)

x = torch.randn(32, 128)
bias = torch.randn(128)
y = x + bias           # bias is treated as (1, 128), broadcasts across batch
```

Broadcasting is fast and idiomatic — use it whenever you would otherwise tile a tensor.

### Automatic differentiation — `requires_grad`, `.backward()`, `.grad`

The whole reason PyTorch exists is to compute gradients automatically. You opt in by setting `requires_grad=True` on a tensor, then compute a scalar loss and call `.backward()`:

```python
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1     # = 16 at x=3
y.backward()
print(x.grad)              # tensor([8.]) — dy/dx = 2x + 2 = 8
```

Three things just happened:

1. PyTorch tracked every operation involving `x` and built a *computational graph*.
2. `.backward()` walked that graph in reverse, applying the chain rule.
3. The result landed in `x.grad`.

### `torch.no_grad()` and `model.eval()`

When you are *evaluating* (validation, inference) you do not want PyTorch to record the graph — it wastes memory and time. Wrap the eval code in `torch.no_grad()`:

```python
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        ...
```

Two different things are happening:

- `model.eval()` flips layers like `Dropout` and `BatchNorm` into eval behavior.
- `torch.no_grad()` turns off the autograd graph entirely.

You need both during validation. Forgetting either is a top-5 cause of "my validation looks weird".

### Detaching from the graph — `.detach()`

When you take a tensor *out* of the graph (for logging, plotting, saving), detach it:

```python
loss_value = loss.detach().cpu().item()
```

This makes a copy that does not require grad and lives on CPU as a Python float — safe to put in a list, write to a file, or pass to `print`.

## Common pitfalls

- `RuntimeError: Expected all tensors to be on the same device` → mix of CPU and GPU tensors → move every operand to `device` before combining.
- Gradient is `None` → forgot `requires_grad=True`, or applied a non-differentiable op (`int` cast, `argmax`, masking) somewhere upstream.
- Gradients pile up across steps → forgot `optimizer.zero_grad()` → call `zero_grad()` at the top of every training step.
- Memory keeps growing during a long evaluation loop → forgot `torch.no_grad()` → wrap the eval loop.
- `view` raises a contiguity error → the tensor is non-contiguous after a `permute` → use `.contiguous().view(...)` or just `.reshape(...)`.

## Learning outcomes

- Create 1-D, 2-D, and 4-D tensors and read off their shape, dtype, and device.
- Move tensors and models between CPU and GPU when CUDA is available.
- Compute a gradient with `autograd` and cross-check it against a hand-derived gradient on a 1-variable function and a 2-variable function.
- Tell the difference between `model.eval()` and `torch.no_grad()` and name what each one switches off.

## Quick check (self-test)

<details>
<summary>Q1 — What is the difference between <code>view</code> and <code>reshape</code>?</summary>

`view` returns a new tensor that shares storage and requires the input to be contiguous. `reshape` may return either a view or a copy and works on non-contiguous tensors.
</details>

<details>
<summary>Q2 — You set <code>requires_grad=True</code>, run <code>y = x.sum()</code> and <code>y.backward()</code>. What does <code>x.grad</code> look like?</summary>

`x.grad` has the same shape as `x` and is filled with `1`s — because `dy/dx_i = 1` for every `i` when `y = sum(x)`.
</details>

<details>
<summary>Q3 — During validation, what is the difference between calling <code>model.eval()</code> and wrapping the loop in <code>with torch.no_grad():</code>?</summary>

`model.eval()` changes the behavior of layers like Dropout (off) and BatchNorm (use running stats). `torch.no_grad()` disables the autograd graph. You usually need both.
</details>

<details>
<summary>Q4 — Shape <code>(32, 3, 224, 224)</code> — what is this?</summary>

A batch of 32 RGB images, each 224 × 224. The dimension order is `(B, C, H, W)`.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 4 (numerical computation), Chapter 6 (feed-forward networks).
- PyTorch official tutorial — *Learn the Basics → Tensors* and *Autograd Mechanics*.
- *PyTorch internals*, blog post by Edward Z. Yang, for the curious.

## Companion artifact

`notebooks/chapter_01_tensor_autograd.ipynb` — every snippet above runs there, plus an autograd-vs-hand-derivative cross-check at the bottom.
