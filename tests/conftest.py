"""Shared pytest fixtures + sys.path injection.

The repo's `src/` is not a proper Python package (no `setup.py`), so we
prepend it to `sys.path` here. Every test file can then do
``import datasets, models, train_text`` directly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


import pytest  # noqa: E402  (must come after sys.path injection)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
