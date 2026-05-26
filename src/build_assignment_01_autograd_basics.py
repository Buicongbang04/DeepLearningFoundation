"""Builder for ``assignments/assignment_01_autograd_basics.ipynb``.

Six guided exercises covering tensor creation, reshaping, broadcasting, and
manual-vs-autograd gradients. The shipped notebook executes cleanly with the
reference solutions in place. Students are expected to delete each solution
body, re-implement it, and verify the assertions still pass.
"""

from _nbutils import ROOT, code, md, write_notebook


def build_cells():
    return [
        md(
            """
# Assignment 01 — Autograd Basics

This is a graded exercise paired with `docs/01_tensor_autograd_pytorch.md` and
`notebooks/chapter_01_tensor_autograd.ipynb`. There are six small exercises;
each one has a stated goal, a function stub with a reference solution, and an
assertion block that verifies your answer.

**How to use this notebook.**

1. Run all cells once to confirm the reference solutions pass.
2. For each exercise, delete the body of the function (everything between the
   `# ---- YOUR SOLUTION BELOW ----` and `# ---- END ----` markers).
3. Re-implement the function from memory.
4. Re-run the cell. The assertions should still pass.

Estimated time: 45–60 minutes.
"""
        ),
        md("## Setup"),
        code(
            """
import math
import torch
import numpy as np

torch.manual_seed(0)
print("torch", torch.__version__)
"""
        ),
        md(
            """
## Exercise 1 — Build the four canonical shapes

Write a function that returns four tensors of the shapes you will see again
and again in this course:

- `labels` shape `(B,)` — random class labels in `[0, num_classes)`.
- `vectors` shape `(B, D)` — standard-normal vectors.
- `images` shape `(B, C, H, W)` — standard-normal image batch.
- `seqs` shape `(B, T, D)` — standard-normal sequence batch.

Reading off shapes is the most common skill in PyTorch.
"""
        ),
        code(
            """
def make_canonical_shapes(B=8, D=16, C=3, H=32, W=32, T=10, num_classes=10):
    # ---- YOUR SOLUTION BELOW ----
    labels  = torch.randint(0, num_classes, (B,))
    vectors = torch.randn(B, D)
    images  = torch.randn(B, C, H, W)
    seqs    = torch.randn(B, T, D)
    # ---- END ----
    return labels, vectors, images, seqs


labels, vectors, images, seqs = make_canonical_shapes()
assert tuple(labels.shape)  == (8,)
assert tuple(vectors.shape) == (8, 16)
assert tuple(images.shape)  == (8, 3, 32, 32)
assert tuple(seqs.shape)    == (8, 10, 16)
assert labels.dtype == torch.int64
assert vectors.dtype == torch.float32
print("ex1 OK")
"""
        ),
        md(
            """
## Exercise 2 — Reshape without copy when possible

Given an input of shape `(B, C, H, W)`, return it flattened to shape
`(B, C*H*W)`. Use the cheapest reshape operation you can.

Hint: right after a `randn`, the tensor is contiguous. Which of `view` /
`reshape` is appropriate?
"""
        ),
        code(
            """
def flatten_image_batch(x):
    # ---- YOUR SOLUTION BELOW ----
    return x.view(x.size(0), -1)
    # ---- END ----


x = torch.randn(4, 3, 8, 8)
y = flatten_image_batch(x)
assert tuple(y.shape) == (4, 192)
assert y.data_ptr() == x.data_ptr(), "view should share storage with x"
print("ex2 OK")
"""
        ),
        md(
            """
## Exercise 3 — Broadcast a per-channel bias

Given an image batch `x` of shape `(B, C, H, W)` and a 1-D `bias` of shape
`(C,)`, add the bias to every spatial location of every image, per channel.

You should not write a `for` loop. Use broadcasting.
"""
        ),
        code(
            """
def add_channel_bias(x, bias):
    # ---- YOUR SOLUTION BELOW ----
    return x + bias.view(1, -1, 1, 1)
    # ---- END ----


B, C, H, W = 2, 3, 4, 4
x = torch.zeros(B, C, H, W)
bias = torch.tensor([1.0, 10.0, 100.0])
y = add_channel_bias(x, bias)
assert tuple(y.shape) == (B, C, H, W)
assert torch.allclose(y[:, 0], torch.ones(B, H, W))
assert torch.allclose(y[:, 1], 10 * torch.ones(B, H, W))
assert torch.allclose(y[:, 2], 100 * torch.ones(B, H, W))
print("ex3 OK")
"""
        ),
        md(
            """
## Exercise 4 — Autograd on a polynomial

Compute the gradient of `f(x) = 3*x^3 - 5*x^2 + 7*x - 2` at `x = 1.5`.

Hand derivation: `f'(x) = 9*x^2 - 10*x + 7`. At `x = 1.5`, `f'(1.5) = 12.25`.
"""
        ),
        code(
            """
def grad_of_polynomial(x_value: float) -> float:
    # ---- YOUR SOLUTION BELOW ----
    x = torch.tensor([x_value], requires_grad=True)
    f = 3 * x ** 3 - 5 * x ** 2 + 7 * x - 2
    f.backward()
    return x.grad.item()
    # ---- END ----


g = grad_of_polynomial(1.5)
print(f"f'(1.5) = {g:.4f}  (expected 12.25)")
assert math.isclose(g, 12.25, abs_tol=1e-4)
print("ex4 OK")
"""
        ),
        md(
            """
## Exercise 5 — Manual gradient of a 2-input function

Function: `g(a, b) = a*b + log(a) + exp(b)` at `(a, b) = (2.0, 0.5)`.

Hand derivation:

- `dg/da = b + 1/a`        → at `a=2, b=0.5` → `0.5 + 0.5 = 1.0`
- `dg/db = a + exp(b)`     → at `a=2, b=0.5` → `2.0 + e^{0.5} ≈ 3.6487`

Compute both gradients via autograd and assert they match the hand values.
"""
        ),
        code(
            """
def autograd_two_variable():
    # ---- YOUR SOLUTION BELOW ----
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([0.5], requires_grad=True)
    g = a * b + torch.log(a) + torch.exp(b)
    g.backward()
    return a.grad.item(), b.grad.item()
    # ---- END ----


da, db = autograd_two_variable()
expected_da = 0.5 + 1 / 2.0
expected_db = 2.0 + math.exp(0.5)
print(f"dg/da = {da:.6f}  (expected {expected_da:.6f})")
print(f"dg/db = {db:.6f}  (expected {expected_db:.6f})")
assert math.isclose(da, expected_da, abs_tol=1e-5)
assert math.isclose(db, expected_db, abs_tol=1e-5)
print("ex5 OK")
"""
        ),
        md(
            """
## Exercise 6 — Gradient accumulation and `zero_grad`

Run the same forward+backward *twice* in a row without zeroing `.grad`
between calls. Observe that the second `.grad` is *twice* the first — the
gradients add up.

Then re-do the experiment with `x.grad = None` (or `.grad.zero_()`) in the
middle, and confirm the gradient is the same as a single backward.

This is exactly why the training loop calls `optimizer.zero_grad()` at the
top of every step.
"""
        ),
        code(
            """
def gradient_accumulation_demo():
    x = torch.tensor([2.0], requires_grad=True)

    # First backward.
    (x ** 2).backward()
    grad_after_first = x.grad.item()

    # Second backward, no zero in between.
    (x ** 2).backward()
    grad_after_second = x.grad.item()

    # Reset and third backward.
    x.grad = None
    (x ** 2).backward()
    grad_after_reset = x.grad.item()

    return grad_after_first, grad_after_second, grad_after_reset


a, b, c = gradient_accumulation_demo()
print(f"after 1st backward : {a}  (expected 4.0)")
print(f"after 2nd backward : {b}  (expected 8.0  — accumulated)")
print(f"after zero + back  : {c}  (expected 4.0  — back to one backward)")
assert math.isclose(a, 4.0)
assert math.isclose(b, 8.0)
assert math.isclose(c, 4.0)
print("ex6 OK")
"""
        ),
        md(
            """
## Final check

If every `ex_ OK` line is printed above, you have completed the assignment.
Submit this notebook (with your re-implementations) plus a one-paragraph
reflection on which exercise was the trickiest.
"""
        ),
        code(
            """
print("All six exercises completed.")
"""
        ),
    ]


def main():
    cells = build_cells()
    out_path = ROOT / "assignments" / "assignment_01_autograd_basics.ipynb"
    write_notebook(cells, out_path)
    print(f"wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
