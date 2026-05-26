# Contributing

This guide is for anyone extending the course — adding a chapter, a lab, an assignment, or a mini-project.

## TL;DR

- Don't hand-write `.ipynb` files. Write a builder script in `src/build_*.py` and check it in alongside the produced notebook. Notebooks are diff-noisy; builders are not.
- Every new notebook / project / lab should be executable end-to-end on a clean `dlcourse` env: `jupyter nbconvert --to notebook --execute <path>` must finish without errors.
- Generated artifacts (checkpoints, downloaded datasets, run logs, MLflow runs, wandb runs) belong in `.gitignore`, not in the repo. Keep `.gitkeep` files so empty output folders are tracked.
- Every chapter docs file follows `docs/CHAPTER_TEMPLATE.md`. Every project follows the layout in `projects/final_project_template/`.

## Eight-phase build plan

The repo is built in eight phases. `ROADMAP.md` describes the *content*; this section describes the *build order*. Each phase produces a self-contained, runnable slice of the course.

| Phase | Theme                       | Content shipped                                                                                              |
|-------|-----------------------------|--------------------------------------------------------------------------------------------------------------|
| 1     | Repo skeleton               | `README`, `COURSE_OVERVIEW`, `SYLLABUS`, `ROADMAP`, `requirements`, all 17 chapter docs, template files.     |
| 2     | PyTorch foundation          | Tensor, autograd, `Dataset`/`DataLoader`, training loop, checkpoint, simple inference.                       |
| 3     | Core neural network         | MLP, backpropagation, activation, initialization, regularization, optimization.                              |
| 4     | Vision foundation           | CNN, data augmentation, ResNet, transfer learning.                                                           |
| 5     | Sequence and Transformer    | RNN, LSTM, attention, Transformer.                                                                           |
| 6     | Representation and generative | Autoencoder, VAE, GAN, diffusion (concept).                                                                |
| 7     | Engineering quality         | Config system, logging, checkpoint naming, testing, reproducibility, CLI scripts.                            |
| 8     | Final project and polishing | Final-project template, grading rubric, sample report, cleanup, reproducibility check.                       |

A phase is *done* when its slice of `ROADMAP.md` has working notebooks, the relevant project (if any) has a `train.py` that succeeds on a free-tier GPU, and the chapter docs render cleanly.

## Repository layout

```
.
├── README.md, SYLLABUS.md, ROADMAP.md   # learner-facing entry points
├── docs/                                # narrative chapter notes (Markdown)
├── notebooks/                           # one runnable notebook per chapter
├── labs/                                # multi-chapter integration labs
├── assignments/                         # student-graded exercises
├── projects/                            # mini-projects + final_project_template
├── datasets/                            # docs only; raw files are gitignored
├── src/                                 # reusable utilities + builder scripts
├── configs/                             # YAML config files
├── experiments/                         # checkpoints, logs (gitignored)
├── tests/                               # pytest smoke tests
├── figures/                             # generated diagrams shared across chapters
└── reports/                             # optional written experiment reports
```

## Adding a new chapter

1. **Doc.** Copy `docs/CHAPTER_TEMPLATE.md` to `docs/NN_topic.md` and write the narrative. Match the template's headings exactly: *Mục tiêu*, *Why this chapter*, *Key concepts*, *Code snippets*, *Common pitfalls*, *Learning outcomes*, *Quick check*, *Further reading*.
2. **Notebook builder.** Create `src/build_chapter_NN_topic.py` following the pattern in `src/build_chapter_01_tensor_autograd.py`:
   - Use `nbformat.v4` with `md(...)` / `code(...)` helpers.
   - Generate one concept per cell, with a Markdown cell explaining the *why* before each code cell.
   - End with `write_notebook(cells, ROOT / "notebooks" / "chapter_NN_topic.ipynb")`.
3. **Build & verify.**
   ```bash
   conda run -n dlcourse python src/build_chapter_NN_topic.py
   conda run -n dlcourse jupyter nbconvert --to notebook --execute notebooks/chapter_NN_topic.ipynb --output chapter_NN_topic.ipynb
   ```
4. **ROADMAP.** Update the relevant row in `ROADMAP.md` if scope changes.
5. **Commit.** Include the builder, the executed notebook, and any new figures under `figures/` in the same commit.

## Adding a new project

1. Copy `projects/final_project_template/` to `projects/project_NN_<name>/`.
2. Fill in `README.md`, `report.md`, and `configs/<name>.yaml`.
3. Reuse `src/datasets.py`, `src/models.py`, `src/train.py`, `src/evaluate.py`, `src/inference.py` from the repo root whenever possible — only add project-local `src/` modules for genuinely project-specific code.
4. The project must run end-to-end via:
   ```bash
   python src/train.py     --config configs/<name>.yaml
   python src/evaluate.py  --config configs/<name>.yaml --checkpoint experiments/<name>/checkpoints/best.pt
   python src/inference.py --checkpoint experiments/<name>/checkpoints/best.pt --input <sample>
   ```
5. Save metrics via `json.dump(results, "results/metrics.json")` — gitignored, regenerated by `train.py`.
6. Save figures via `fig.savefig("figures/<name>.png", dpi=120)`. The `figures/` folder ships a `.gitkeep`; generated `.png` files are *not* committed.

## Coding standard

- Every model inherits `nn.Module`.
- No hard-coded hyperparameters; load from YAML in `configs/`.
- The training script saves a checkpoint by validation metric and logs every epoch.
- `train.py`, `evaluate.py`, `inference.py` are separate files with `argparse` CLI.
- Notebooks are for learning and demos; scripts in `src/` are for serious work.
- Type hints encouraged on `src/` helpers; not required inside notebooks.

## Reproducibility standard

Every project ships with:

- `config.yaml` (or `configs/<name>.yaml`) — everything that determines a run.
- A fixed `seed` field in the config.
- The version of every key package (`requirements.txt` is enough at the repo root).
- A README with the *exact* commands to train, evaluate, and run inference.
- The dataset split documented (sizes, source link, preprocessing).
- The best-checkpoint metric logged in `report.md`.

## Testing standard

Minimum `tests/` to keep green:

- `tests/test_dataset.py` — at least one batch loads with the expected shape.
- `tests/test_model.py` — forward pass on a dummy batch returns the expected output shape.
- `tests/test_loss_metric.py` — loss and metric functions return a scalar for a dummy batch.
- `tests/test_config.py` — every YAML in `configs/` parses.
- Test runner: `pytest tests/`.

## Style

- **Code.** Black-ish formatting (4-space indent, no trailing whitespace).
- **Notebooks.** One concept per cell. Markdown cells explain the *why* of the next code cell. No silent re-imports across cells.
- **Comments.** Prefer well-named variables over comments. Comment only when the *why* is non-obvious.
- **Docs.** Follow the headings in `docs/CHAPTER_TEMPLATE.md`. Keep narrative short and link out to the original references.

## .gitignore policy

Anything generated by running code is gitignored. Concretely:

- `experiments/**` — checkpoints, run logs, MLflow/wandb runs.
- `projects/*/results/*.json`, `projects/*/results/*.csv` — outputs of `train.py`.
- `projects/*/figures/*.png` — plots saved by notebooks. (If a figure is conceptually static and belongs in the docs, copy it to `figures/` at the repo root and commit it there.)
- `datasets/*.{csv,zip,tar.gz,parquet,npy,pt}` — keep only `datasets/README.md` describing where the data comes from.
- `DeepLearningBook.pdf`, `Instruction.pdf`, any other course PDF — never commit.

When you add a new producer (a notebook or script that writes to disk), add the matching pattern to `.gitignore` in the same commit.

## Running everything locally

```bash
conda env create -f environment.yml
conda activate dlcourse

# Build any notebook from its source script.
python src/build_chapter_01_tensor_autograd.py

# Re-execute (this is also the test).
jupyter nbconvert --to notebook --execute notebooks/chapter_01_tensor_autograd.ipynb --output chapter_01_tensor_autograd.ipynb

# Train a mini-project from CLI.
python src/train.py --config configs/mlp_mnist.yaml

# Run the smoke tests.
pytest tests/
```

## Pull-request checklist

- [ ] New / modified notebooks execute cleanly with `jupyter nbconvert --execute`.
- [ ] Builder scripts in `src/` produce the same notebook bit-for-bit (don't hand-edit the `.ipynb`).
- [ ] New disk writes have matching entries in `.gitignore`.
- [ ] `ROADMAP.md` reflects the new content's status.
- [ ] `pytest tests/` is green.
- [ ] No `DeepLearningBook.pdf`, `Instruction.pdf`, or large dataset files staged.
