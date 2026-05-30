# Project-local `src/`

Two rules:

1. **Reuse the repo-root `src/` whenever you can.** The repo-root
   `src/train.py`, `src/evaluate.py`, `src/inference.py`,
   `src/datasets.py`, `src/models.py` are the shared training stack
   and they accept any YAML config.
2. **Only add files here for genuinely project-specific code** — a
   custom `Dataset`, a custom `nn.Module`, a post-training script that
   the shared driver does not need to know about.

If you find yourself wanting to *override* a file in the shared
`src/`, propose the change at the shared level instead — the per-project
folder is for *additions*, not replacements.
