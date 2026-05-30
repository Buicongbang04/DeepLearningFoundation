# Final Project — Grading Rubric

A single grading sheet used to evaluate the Chapter-17 final project.
**100 points total.** Pair this with `docs/16_final_project.md` (scope
guidance) and `projects/final_project_template/` (starter layout).

Each row lists *what* is being graded, *how many points* it is worth,
and the exact thing the grader will look for.

## 1. Scope and problem framing — 10 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| 10/10 | Problem stated in one paragraph: task, dataset, metric. All three are well-defined and tractable in the project's time budget. |
|  7/10 | Problem stated but metric or dataset choice is fuzzy (e.g., "we want the model to do well"). |
|  4/10 | Problem is over-scoped ("beat SOTA on ImageNet") or under-defined (no metric stated).        |
|  0/10 | Problem is missing or the dataset/metric are inconsistent.                                   |

## 2. Method — 10 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| 10/10 | Architecture in one sentence + training recipe (optimizer, LR, schedule, epochs, batch size, seed). Each piece cites the chapter or paper it comes from. Reuses repo-root `src/` where possible. |
|  7/10 | Recipe is present but missing one item (e.g., seed not pinned, schedule not described).      |
|  4/10 | Architecture or recipe is opaque — "we used a Transformer" with no details.                  |
|  0/10 | No method section, or it contradicts the configs/scripts in the repo.                        |

## 3. Baseline — 10 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| 10/10 | A trivial baseline (Logistic Regression, KNN, "majority class", or a known weaker architecture) is trained and reported alongside the main model. The main model clearly beats it. |
|  7/10 | Baseline is mentioned but only as "we tried LR and it was bad" without a concrete metric.    |
|  0/10 | No baseline — headline metric is unanchored.                                                 |

## 4. Results table and training curve — 15 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| 15/15 | Table of train/val (and ideally test) metric per epoch, plus the headline metric with the best-epoch marked. Training-curve figure (`figures/training_curve.png`) embedded in the report. The figure shows train **and** val on the same axes and the best epoch is highlighted. |
| 10/15 | Headline metric is present but no per-epoch table, or the training curve is missing the best-epoch marker. |
|  5/15 | Only a single headline number with no curve.                                                  |
|  0/15 | No quantitative results at all.                                                              |

## 5. Reproducibility — 15 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| 15/15 | Three commands (`train.py`, `evaluate.py`, `inference.py`) listed verbatim; `evaluate.py` reproduces the headline metric exactly; `seed` is pinned; `requirements.txt` works; the report shows the exact reported metric ± expected noise. |
| 10/15 | Commands run but reproduce a slightly different number (within 0.5 percentage points) — usually a forgotten seed step. |
|  5/15 | Commands listed but at least one fails on a fresh env.                                       |
|  0/15 | No reproducibility section, or hand-edited checkpoint values.                                |

## 6. Error analysis — 15 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| 15/15 | Confusion matrix (or analogue), per-class metric breakdown, and a grid of "got it wrong" examples with a one-paragraph interpretation. |
| 10/15 | Confusion matrix + per-class breakdown but no qualitative failure examples.                  |
|  5/15 | Only a confusion matrix, no interpretation.                                                  |
|  0/15 | Only the aggregate metric — no localization of what the model gets wrong.                    |

## 7. "What I would try next" — 10 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| 10/10 | *One* concrete experiment with a hypothesis and an expected delta from the current result.   |
|  6/10 | A wishlist of 3-5 future ideas, none committed to.                                          |
|  3/10 | One vague item ("more data would probably help").                                            |
|  0/10 | Section is missing.                                                                          |

## 8. Code quality — 10 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| 10/10 | All code passes `pytest tests/`; project-specific code lives under `projects/final_<your>/src/`; the shared `src/` is reused where possible; YAML config has no hard-coded paths. |
|  6/10 | Tests pass but configuration leaks (hard-coded `cuda` device, hard-coded paths) make the project non-portable. |
|  3/10 | Tests fail or there is no `configs/main.yaml`.                                              |
|  0/10 | Code does not run, or there is no project-local code at all.                                 |

## 9. Report quality — 5 points

| Score | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
|  5/5  | `report.md` follows the template; 2-4 pages; sentences are tight; tables and figures are referenced from the prose.                          |
|  3/5  | Report exists but has irrelevant filler or skips one template section.                       |
|  0/5  | Report missing or unintelligible.                                                            |

## Bonus (cap +10, but final score still 100 max)

| Bonus | Description                                                                                  |
|------:|----------------------------------------------------------------------------------------------|
| +5    | A clearly-explained negative result ("I expected X, got Y, here is why I think it failed"). Honesty is rewarded.           |
| +5    | Ablation beyond the baseline + main model — e.g., turning off augmentation, swapping the optimizer, varying capacity. |

## Anti-bonuses (deductions)

- **−5** for any committed file that should be gitignored (a `.pt`
  checkpoint, a downloaded dataset, a generated `figures/*.png`).
- **−5** for committed PDF reference material (e.g.,
  `DeepLearningBook.pdf`).
- **−10** for hand-edited metrics in the report that do not match the
  re-evaluated checkpoint.

## How to use this rubric

1. Read the report end-to-end before clicking around the repo.
2. Run the three commands from the "Reproducibility" section. The
   headline metric must match.
3. Run `pytest tests/`. The suite must be green.
4. Score row by row. Sum.
