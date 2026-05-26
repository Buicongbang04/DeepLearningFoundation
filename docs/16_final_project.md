# Chapter 17 — Final Project: Deep Learning End-to-End

> Up to here, every chapter gave you one tool. The final project tests whether you can pick the *right* tools for an actual problem of your choice, glue them together, and deliver a repo that someone else can clone, install, train, and re-evaluate without asking you a single question.

## Mục tiêu (Goal)

After this chapter you can:

- Scope a Deep Learning problem to something you can actually finish in 2–4 weeks of part-time work.
- Build a baseline, a main model, a training pipeline, an evaluation script, and an inference script — all in one self-contained repo.
- Write a report that shows a training curve, an experiment comparison, an error analysis, and one direction for improvement.
- Ship the project with reproducibility instructions clear enough that a stranger can re-run the best result in one command.

## Why this chapter

The 16 previous chapters were structured, with one concept per chapter and a worked example each. The final project intentionally has no worked example. The skill being assessed is *integration* — knowing which chapter's tool to reach for at each step, in the right order, on a problem the course did not anticipate.

- **Builds on:** *every* previous chapter. In practice you will reach for Ch 4 (training loop), Ch 7 (optimizer), Ch 9 or 12 (architecture), Ch 13 (debugging), Ch 16 (inference).
- **Sets up:** the rest of your Deep Learning career. The patterns in this template (config-driven training, CLI inference, README usage block) are the patterns of every serious DL repo.

## Key concepts

### Scoping — pick a problem you can finish

The single most common failure mode of a final project is *scope*. Choose a problem where:

- A labeled dataset already exists and fits on your disk.
- The baseline is a known architecture from this course (MLP, CNN, LSTM, small Transformer, autoencoder).
- The metric is well-defined (accuracy, F1, mean reconstruction MSE, perplexity).
- Training a usable model takes minutes-to-hours on the hardware you have — not days.

Examples that fit the budget:

- Image classification on a small custom dataset of 5–20 classes, 100–1000 images each, via transfer learning (Ch 9).
- Text classification on a public dataset (AG News, IMDB) with an LSTM or small fine-tuned Transformer.
- Anomaly detection on a tabular or image dataset using an autoencoder (Ch 14).
- A toy character-level language model on a single book or a small chat log (Ch 12).

Examples that do *not* fit and you should refuse:

- "Train a chatbot like ChatGPT" (months of compute, billions of tokens).
- "Diagnose cancer from CT scans" (data access, regulatory, ethical concerns out of scope).
- "Beat state of the art on ImageNet" (you cannot in one project, and the time is better spent elsewhere).

### The repo layout

Use `projects/final_project_template/` as your starting point. Concretely:

```
projects/final_<your_name>/
├── README.md              # what the project does + usage
├── report.md              # the written report (see below)
├── configs/
│   └── main.yaml          # the one config that runs the headline result
├── src/
│   ├── datasets.py        # custom Dataset class
│   ├── models.py          # model definition
│   ├── train.py           # CLI: trains, saves best checkpoint
│   ├── evaluate.py        # CLI: loads checkpoint, reports val/test metrics
│   └── inference.py       # CLI: predicts on new input
├── experiments/           # gitignored: checkpoints, logs
└── figures/               # generated: training curves, sample grids, confusion matrix
```

The CLI scripts mirror the recipes in Ch 4 and Ch 16. Reuse `src/train.py`, `src/evaluate.py`, `src/inference.py` from the repo root whenever they fit; only override locally if your project genuinely needs something different.

### The baseline–main–ablation arc

A serious project compares at least three configurations:

1. **Baseline** — the simplest model that solves the problem at all. For image classification, that is "Logistic Regression on flattened pixels" or "MLP-only". This sets the floor.
2. **Main model** — your headline architecture (CNN, LSTM, small Transformer, pretrained backbone). This is what the report's title is about.
3. **At least one ablation** — turn off one component and re-train. Examples: with vs. without data augmentation, with vs. without dropout, frozen vs. fine-tuned backbone.

The ablation is what shows *why* your main result is better, not just *that* it is.

### Training curve — what to plot and why

The headline figure of the report is a train/val loss curve, with the best-val checkpoint marked. Generate it directly from your logged metrics:

```python
import matplotlib.pyplot as plt
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"],   label="val")
plt.axvline(best_epoch, linestyle="--", color="gray", label=f"best @ epoch {best_epoch}")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.savefig("figures/training_curve.png", dpi=120)
```

If you cannot tell, from looking at this plot, whether the model is under- or over-fitting, re-read Chapter 13 before submitting. The plot is also the most-asked question in any presentation — be ready to explain every kink.

### Error analysis — beyond aggregate metrics

A single accuracy number hides everything that matters. Add:

- **Confusion matrix** (for classification) — `sklearn.metrics.confusion_matrix(y_true, y_pred)` plus a `ConfusionMatrixDisplay`. Look for any one cell with most of the errors.
- **Per-class metric breakdown** — precision/recall per class (`classification_report`). The worst class often tells you what is going wrong (under-represented label, ambiguous category).
- **A grid of "model got it wrong" examples** — sample 16 misclassified examples and put them in a 4×4 grid. Half of all dataset bugs reveal themselves in this grid.
- **One success and one failure mode in plain prose** — "the model is good at X, bad at Y; the most common Y → Z confusion is probably because …". This sentence is what graders read first.

### Reproducibility — the one-command standard

A reproducible project means a stranger can run *three* commands and get your headline result:

```bash
pip install -r requirements.txt
python src/train.py    --config configs/main.yaml
python src/evaluate.py --config configs/main.yaml --checkpoint experiments/checkpoints/best.pt
```

For that to work, the config must contain:

- The exact `seed` (set in `train.py` via `torch.manual_seed`, `random.seed`, `np.random.seed`).
- The data source — either an automatic `torchvision`/`huggingface` download or a documented URL with checksums.
- The architecture hyperparameters and the optimizer/scheduler config.
- The metric reported in the report and how it is computed.

`pip freeze > requirements.txt` is fine as a fallback if your top-level `requirements.txt` is too coarse for the exact versions you used.

### The report — what graders look for

A 2–4 page Markdown report in `report.md`, structured as:

1. **Problem** — one paragraph: what task, what dataset, what metric.
2. **Method** — one paragraph: baseline, main model, training recipe. Cite the chapter or paper each piece comes from.
3. **Results** — table of metrics for baseline / main / ablation. Training-curve figure. One paragraph on what surprised you.
4. **Error analysis** — confusion matrix or analogue, per-class breakdown, "got-it-wrong" examples.
5. **What I would try next** — *one* concrete next experiment with a hypothesis (not a wishlist).
6. **Reproducibility** — the three commands above, plus seed, plus expected metric ± noise.

Tighter is better. A reviewer should be able to tell whether the project is good in two minutes; the rest of the read is verification.

### Limits — what to be honest about

A good report is honest about what the project does *not* do:

- "Accuracy is 88%, but only on a clean validation set; in the wild the input distribution will shift."
- "Inference takes 200 ms per image on CPU; a real product would need TorchScript export or ONNX (Ch 16)."
- "The dataset is class-imbalanced 4:1; future work should try focal loss or stratified augmentation."

This is the difference between a course project and an over-confident demo.

## Common pitfalls

- Project chosen on Friday, due on Monday → not enough time to iterate → submit a half-broken pipeline. Pick the problem in the first week.
- Headline metric reported from a *training* epoch, not a held-out validation/test split → leakage; numbers are bogus. Always evaluate on a split the model never saw.
- Saved only `last.pt` → metrics in the report are worse than what was achievable. Save `best.pt` by validation metric (Ch 16).
- README does not include the three reproducibility commands → grader cannot run your project → score collapses. Write the README *first*, then code to match it.
- No baseline → cannot say whether your main model is actually better than a Logistic Regression. Always add a trivial baseline.
- Single run, no seed control → next person sees a different number → suspicion of cherry-picking. Set a seed, optionally average over 3 seeds for the headline.
- "What I would try next" is a wishlist of 10 items → reads as not committed. Pick *one* concrete experiment with a hypothesis.

## Learning outcomes

- Scope a Deep Learning project to a clear problem, dataset, baseline, main model, and metric.
- Produce a repo that lives entirely in `projects/final_<your_name>/` and runs end-to-end via `train.py`, `evaluate.py`, `inference.py`.
- Plot a clean train/val curve and read it as a diagnostic.
- Write a 2–4 page report with results, error analysis, and one next-step proposal.
- Deliver reproducibility instructions that a stranger can follow in three commands.

## Quick check (self-test)

<details>
<summary>Q1 — Why is a *baseline* required in addition to the main model?</summary>

To know whether your main model is actually better than a trivial alternative. A 92%-accurate Transformer on a problem where Logistic Regression hits 91% is not a win. Without a baseline, the headline number is unanchored and the experiment is incomplete.
</details>

<details>
<summary>Q2 — Your report cites 95% accuracy but the grader runs `evaluate.py` and gets 91%. Two likely causes?</summary>

You loaded the *last* checkpoint instead of the best one, *or* you reported a train/val metric in the report but the grader is running on test. Both are recoverable by saving `best.pt` (Ch 16) and being explicit about which split each number comes from.
</details>

<details>
<summary>Q3 — What goes in "What I would try next" — a wishlist or a hypothesis?</summary>

One concrete experiment with a hypothesis. Example: "the worst class is C, which is also the smallest in the training set; I would try weighted sampling or focal loss and expect the macro-F1 to improve by ≥ 2 points." A wishlist of 10 items reads as not having actually thought it through.
</details>

<details>
<summary>Q4 — Name three things that must be inside the config to make the run reproducible.</summary>

The random seed; the exact architecture hyperparameters; the optimizer + scheduler config (including LR, weight decay, schedule). Also helpful: the data source URL or `torchvision`/`huggingface` identifier, and the requirement-pinned package versions.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 12 (applications), as a survey of what kinds of problems are tractable.
- Andrej Karpathy, *A Recipe for Training Neural Networks* (blog) — the methodology backbone for any final project.
- Sebastian Raschka, *Machine Learning Q and AI* — chapters on model evaluation and reproducibility.
- The README of any well-maintained PyTorch repo (e.g., `timm`, `nanoGPT`) — as a model for what a "good README" looks like.

## Companion artifact

`projects/final_project_template/` — the starter layout with placeholder `README.md`, `report.md`, `configs/main.yaml`, and stub `src/` files for `datasets.py`, `models.py`, `train.py`, `evaluate.py`, `inference.py`.
