# Chapter 2-3 — From Logistic Regression to MLP, with Backpropagation

> Logistic regression is a 1-layer neural network with no hidden layer. A *multi-layer* perceptron (MLP) adds a hidden layer with a non-linearity — and that single change unlocks everything from XOR up to deep networks. This chapter spans Chapters 2 and 3 of the roadmap: Part A goes from logistic regression to MLP; Part B walks backpropagation by hand and then with `autograd`.

## Mục tiêu (Goal)

After this chapter you can:

- Explain why no linear model can solve XOR, and why a single hidden layer with a non-linear activation can.
- Build and train a 2-input, 1-hidden-layer MLP in PyTorch on a 2-D toy dataset, and draw the decision boundary.
- Write the forward pass and backward pass of a small MLP *by hand* in NumPy.
- Cross-check the hand-derived gradients against PyTorch's `autograd`.

## Why this chapter

You probably already trained logistic regression in your intro-ML course. This chapter explains why that one trick — adding a hidden layer with a non-linear activation — fundamentally changes what the model can express, and then forces you to do backpropagation *by hand* once, so the rest of the course can use `autograd` confidently.

- **Builds on:** Chapter 1 (tensor, autograd).
- **Sets up:** Chapter 4 (standard training loop), Chapter 5 (activation/init/normalization choices), Chapter 6 (regularization), every later chapter.
- **External prerequisite:** chain rule from calculus, matrix multiplication, sigmoid and softmax from intro ML.

---

# Part A — From logistic regression to MLP

## Linear model recap

A linear model:

```
z = W·x + b
ŷ = σ(z)        # sigmoid for binary classification
```

For binary classification with binary cross-entropy loss:

```
L = -[ y · log(ŷ) + (1 - y) · log(1 - ŷ) ]
```

`W` is a `(D,)` vector, `b` is a scalar (binary), or `W ∈ R^{D×K}`, `b ∈ R^K`, ŷ goes through softmax (multi-class). In PyTorch:

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(2, 1))   # 2 input features, 1 output logit
loss_fn = nn.BCEWithLogitsLoss()         # combines sigmoid + BCE for numerical stability
```

This is logistic regression. There is *no* hidden layer.

## The XOR problem

Consider the four points:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

The decision boundary of any linear model is a *straight line* in 2-D. There is no single straight line that separates the `y=1` points from the `y=0` points here. Try a few and you will see — XOR is not linearly separable.

This is not a curiosity; it generalized into a 1969 critique that paused neural network research for over a decade. The way out turns out to be embarrassingly simple.

## The perceptron and the MLP

A **perceptron** is one neuron: linear combination, then a step function (or sigmoid).

A **multi-layer perceptron (MLP)** stacks two or more linear-then-non-linear layers:

```
h = σ_1( W_1 · x + b_1 )      # hidden layer
ŷ = σ_2( W_2 · h + b_2 )      # output layer
```

Without `σ_1` (the non-linearity), the whole thing collapses back into a single linear model:

```
W_2 · (W_1 · x + b_1) + b_2 = (W_2 · W_1) · x + (W_2 · b_1 + b_2) = W' · x + b'
```

It is the **non-linear activation** between layers that gives MLPs their power. Common choices:

- **ReLU** — `max(0, z)`. Default for hidden layers.
- **Sigmoid** — `1 / (1 + e^{-z})`. Default for binary output.
- **Tanh** — `(e^z - e^{-z}) / (e^z + e^{-z})`. Older default for hidden layers.
- **Softmax** — multi-class output normalization.

## Solving XOR with a 2-2-1 MLP

A 2-input, 2-hidden, 1-output MLP with ReLU solves XOR comfortably:

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)        # logit
```

After ~1000 SGD steps on the four XOR points, the decision boundary becomes a *piecewise-linear* curve that separates the four corners. The hidden layer has, in effect, learned a non-linear feature transformation that makes the problem linearly separable in `h` space — even though it is not in `x` space.

## Decision boundary

You can plot the decision boundary by running the model on a fine grid of input points and coloring by predicted class. The XOR notebook does this. The takeaway is visual: the boundary bends. A linear model could not produce that bend.

---

# Part B — Forward pass, loss, backpropagation

## One training step, in words

For one example `(x, y)`:

1. **Forward pass.** Compute `ŷ = f(x; θ)` and a loss `L = loss(ŷ, y)`.
2. **Backward pass.** Compute `∂L/∂θ` for every parameter `θ` (every entry of every weight matrix and bias vector).
3. **Update.** `θ ← θ - η · ∂L/∂θ` for some learning rate `η`.

Backpropagation is the algorithm for step 2. It is the chain rule from calculus, applied node-by-node through the computational graph.

## Forward pass for a 2-2-1 MLP, by hand

Let:

```
x  ∈ R^2,    W_1 ∈ R^{2×2}, b_1 ∈ R^2,    W_2 ∈ R^{1×2}, b_2 ∈ R^1
z_1 = W_1 · x + b_1         # (2,)
h   = σ(z_1)                 # (2,)  — pick ReLU or sigmoid; ReLU here
z_2 = W_2 · h + b_2          # (1,)
ŷ   = σ(z_2)                 # (1,)  — sigmoid
L   = -[ y log ŷ + (1-y) log(1-ŷ) ]
```

You can compute every quantity above on paper given a specific `x`, `W_1`, `W_2`, `b_1`, `b_2`. Doing it once builds intuition.

## Chain rule and gradients

We want `∂L/∂W_1`, `∂L/∂W_2`, `∂L/∂b_1`, `∂L/∂b_2`. By the chain rule, work backward from `L`:

```
∂L/∂ŷ   = -(y/ŷ) + (1-y)/(1-ŷ)        (gradient of BCE w.r.t. ŷ)
∂ŷ/∂z_2 = ŷ · (1 - ŷ)                  (sigmoid derivative)
∂L/∂z_2 = (∂L/∂ŷ) · (∂ŷ/∂z_2)
        = ŷ - y                        (nice simplification for sigmoid + BCE)
```

Then propagating into `W_2`, `b_2`, `h`, `z_1`, `W_1`, `b_1`:

```
∂L/∂W_2 = (∂L/∂z_2) · h^T                  # (1, 2)
∂L/∂b_2 = ∂L/∂z_2                          # (1,)
∂L/∂h   = W_2^T · (∂L/∂z_2)                # (2,)
∂L/∂z_1 = (∂L/∂h) ⊙ relu'(z_1)             # (2,) — element-wise
∂L/∂W_1 = (∂L/∂z_1) · x^T                  # (2, 2)
∂L/∂b_1 = ∂L/∂z_1                          # (2,)
```

`relu'(z) = 1 if z > 0 else 0`. `⊙` is element-wise product.

## NumPy implementation of the backward pass

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z):    return np.maximum(0, z)

def forward(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    h  = relu(z1)
    z2 = W2 @ h + b2
    yhat = sigmoid(z2)
    return z1, h, z2, yhat

def backward(x, y, W1, b1, W2, b2):
    z1, h, z2, yhat = forward(x, W1, b1, W2, b2)
    dz2  = yhat - y                        # (1,)
    dW2  = dz2[:, None] * h[None, :]       # (1, 2)
    db2  = dz2                             # (1,)
    dh   = W2.T @ dz2                      # (2,)
    dz1  = dh * (z1 > 0).astype(np.float32)
    dW1  = dz1[:, None] * x[None, :]       # (2, 2)
    db1  = dz1                             # (2,)
    return dW1, db1, dW2, db2
```

## Cross-checking against autograd

The notebook then computes the exact same gradients via PyTorch:

```python
import torch
from torch import nn

x_t  = torch.tensor(x, dtype=torch.float32)
y_t  = torch.tensor([y], dtype=torch.float32)

W1_t = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

z1   = W1_t @ x_t + b1_t
h    = torch.relu(z1)
z2   = W2_t @ h + b2_t
yhat = torch.sigmoid(z2)
loss = nn.functional.binary_cross_entropy(yhat, y_t)
loss.backward()

# Now compare W1_t.grad to the NumPy dW1, etc.
```

The two should match to within floating-point noise (`~1e-6` absolute difference). If they do not, you have a bug — typically a transpose, a missing element-wise product, or a wrong shape.

## Why we do this exactly once

Once you have verified that autograd matches your hand-derived gradients for a 2-2-1 MLP, you have permission to *trust* autograd for the rest of the course. From here on the loop is:

```python
loss = criterion(model(x), y)
loss.backward()
optimizer.step()
```

…and gradients are handled invisibly.

## Common pitfalls

- Loss does not change → check the *forward* pass first: print `yhat` and the loss before debugging the backward.
- Gradients are all zero in the first layer → likely "dead ReLU" — the activation is always 0 so the ReLU derivative is always 0. Use a smaller weight init or LeakyReLU.
- The decision boundary stays a straight line on XOR → you forgot the activation between layers, or the activation is linear.
- `BCELoss` produces NaN → use `BCEWithLogitsLoss` (sigmoid inside, more stable) and feed it raw logits, not sigmoid outputs.
- Shapes mismatch when comparing your NumPy gradient to autograd — autograd uses batch-first shapes, your hand-derived version may be batch-of-1.

## Learning outcomes

- Explain why no linear model can solve XOR.
- Build and train a 2-2-1 MLP in PyTorch and produce a decision-boundary plot.
- Compute the forward pass for a small MLP by hand on a chosen `x`.
- Write a NumPy backward pass for the same MLP and verify against autograd.

## Quick check (self-test)

<details>
<summary>Q1 — What single change to the 2-2-1 architecture makes XOR solvable?</summary>

Inserting a non-linear activation (ReLU, tanh, or sigmoid) between the two linear layers. Without it, the network is mathematically equivalent to a single linear layer.
</details>

<details>
<summary>Q2 — For sigmoid + binary cross-entropy, what is the gradient of the loss with respect to the pre-activation logit <code>z_2</code>?</summary>

`∂L/∂z_2 = ŷ - y`. The sigmoid and BCE derivatives cancel beautifully, which is why this combination is numerically stable.
</details>

<details>
<summary>Q3 — Why use <code>BCEWithLogitsLoss</code> instead of applying a <code>sigmoid</code> and then a <code>BCELoss</code>?</summary>

`BCEWithLogitsLoss` fuses the sigmoid and the loss into a single `log-sum-exp`-based formula that avoids `log(0)` when the logit is very large or very small. It is numerically stable; the two-step version is not.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 6 (feed-forward networks), Section 6.5 (backpropagation).
- Karpathy's *micrograd* walk-through on YouTube — manual backprop on a tiny autograd engine.
- PyTorch tutorial — *Build the Neural Network* and *Automatic Differentiation*.

## Companion artifact

`notebooks/chapter_02_mlp_from_scratch.ipynb` — XOR data, 2-2-1 MLP, decision-boundary plot, NumPy backward, autograd cross-check.
