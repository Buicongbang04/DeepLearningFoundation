# Final Project Template

Copy this folder to `projects/final_<your_name_or_topic>/` and use it
as the starting layout for your capstone. Pair it with **Chapter 17 /
`docs/16_final_project.md`** for scope guidance and what the report
must contain.

## What this template gives you

```
final_project_template/
├── README.md          # this file — overwrite with your project's
├── report.md          # 2-4 page write-up (results + error analysis)
├── configs/
│   └── main.yaml      # the one config that produces the headline result
└── src/
    ├── __init__.py
    ├── datasets.py    # add your custom dataset; reuse repo factories where possible
    ├── models.py      # add your custom model; reuse repo factories where possible
    └── README.md      # tiny note on what's local vs. inherited from /src
```

## Recommended workflow

1. **Scope it.** Pick something you can finish in 2–4 weeks of
   part-time work. Read the "Scoping" section in `docs/16_final_project.md`.
2. **Copy the template.** `cp -r projects/final_project_template
   projects/final_<your_name>/`.
3. **Fill `configs/main.yaml`** with the headline run's hyperparameters
   (seed, optimizer, schedule, dataset paths). One config = one
   reproducible headline result.
4. **Decide what to reuse.** The training driver lives at the repo
   root: `python src/train.py --config projects/final_<your>/configs/main.yaml`
   works out of the box if your dataset and model are registered in
   `src/datasets.py` and `src/models.py`. Only add files under
   `projects/final_<your>/src/` for *project-specific* code.
5. **Train, evaluate, infer.** Reproducibility means three commands:

   ```bash
   python src/train.py     --config projects/final_<your>/configs/main.yaml
   python src/evaluate.py  --config projects/final_<your>/configs/main.yaml \
                           --checkpoint experiments/<your_experiment>/checkpoints/best.pt
   python src/inference.py --checkpoint experiments/<your_experiment>/checkpoints/best.pt \
                           --input  <single file or directory>
   ```

6. **Write the report** in `report.md`. Match the structure laid out in
   `docs/16_final_project.md` (Problem → Method → Results → Error
   analysis → What I would try next → Reproducibility).

## What the grader looks for

- The three commands above run cleanly on a fresh `dlcourse` env.
- `report.md` has a training-curve figure, a confusion matrix or
  per-class breakdown, and a *single* "what I would try next" item with
  a hypothesis (not a wishlist).
- A baseline run (Logistic Regression, MLP, or "majority class") is
  cited so the headline metric is anchored.
- The config has a pinned `seed`, the data source is documented, and
  the package versions match the repo's `requirements.txt`.

## What NOT to include

- Large data files (CSV/parquet/checkpoint). Use `datasets/` and
  `experiments/` — both gitignored.
- Any of the course PDFs.
- Half-implemented features. If something does not work, leave it out
  and mention it in "what I would try next".
