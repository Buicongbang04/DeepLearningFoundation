"""Shared helpers for notebook builder scripts.

Every `src/build_chapter_*.py` and `src/build_assignment_*.py` imports `md`,
`code`, and `write_notebook` from here. The convention from CONTRIBUTING.md is
that notebooks are *generated* from these builders, never hand-edited.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def md(source: str) -> nbformat.NotebookNode:
    """Create a Markdown cell. Leading/trailing whitespace is stripped."""
    return new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbformat.NotebookNode:
    """Create a code cell. Leading/trailing whitespace is stripped."""
    return new_code_cell(source.strip("\n"))


def write_notebook(cells: Iterable[nbformat.NotebookNode], path: Path) -> None:
    """Assemble cells into a notebook and write it to ``path``.

    The notebook is written with Python 3 kernelspec metadata so
    ``jupyter nbconvert --execute`` picks the right interpreter.
    """
    nb = new_notebook(cells=list(cells))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
