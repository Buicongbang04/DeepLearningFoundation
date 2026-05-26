# Chapter NN — &lt;Chapter title&gt;

> One-sentence summary of *what* this chapter is about and *why* the learner is reading it.

## Mục tiêu (Goal)

A 2-4 sentence statement of what the learner will be able to do *after* this chapter. Be concrete: "implement X", "explain why Y", "debug Z". This must align with the **Outcomes** row in `ROADMAP.md` for the same chapter.

## Why this chapter

Where this chapter sits in the course flow, what it depends on, and what depends on it. Use the dependency callouts:

- **Builds on:** Chapter `<prev>` and the specific concepts you need to remember.
- **Sets up:** Chapter `<next>` and the role this chapter plays in later content.
- **External prerequisite:** any non-course knowledge needed (NumPy, calculus, etc.).

## Key concepts

Use H3 (`###`) sections for each concept. For each concept:

1. State the concept in one sentence.
2. Give the mathematical or operational definition — keep notation consistent with `docs/01_tensor_autograd_pytorch.md`.
3. Show the smallest possible PyTorch snippet that makes the concept tangible.
4. Call out the *shape* of every tensor in the snippet.

Example structure:

### Concept 1 name

One-sentence definition.

Mathematical form (if any):

```
<formula>
```

Minimum-viable code:

```python
import torch

x = torch.randn(4, 3)        # shape: (batch=4, features=3)
y = x.sum(dim=1)             # shape: (4,)
```

Shape walk-through:

- `x`: `(4, 3)` — 4 examples, 3 features.
- `y`: `(4,)` — one scalar per example.

### Concept 2 name

…

## Common pitfalls

A bulleted list of mistakes that learners typically make in this chapter. Each item should be of the form *"symptom → cause → fix"*. Examples:

- Loss does not change → forgot `optimizer.zero_grad()` → call `zero_grad()` at the top of each step.
- Validation accuracy is 100% but test accuracy collapses → `model.train()` left on during eval → call `model.eval()` before the validation loop.

## Learning outcomes

A bulleted list that *exactly* mirrors the **Outcomes** row in `ROADMAP.md` for this chapter. The learner can use this list to self-assess.

## Quick check (self-test)

3-5 short questions the learner should be able to answer before moving on. Provide the answer in a `<details>` block so the learner can attempt first.

<details>
<summary>Q1 — &lt;question stem&gt;</summary>

Answer: …
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter &lt;N&gt;, section &lt;X.Y&gt;.
- PyTorch documentation — `torch.nn` link or tutorial link.
- One blog post or YouTube lecture, if the topic has a widely-cited explainer (e.g. Karpathy's "Makemore" series for sequence models).

## Companion artifact

A single line pointing to the runnable artifact in the repo: `notebooks/chapter_NN_*.ipynb`, `projects/project_NN_*/`, or `labs/lab_NN_*.ipynb`.
