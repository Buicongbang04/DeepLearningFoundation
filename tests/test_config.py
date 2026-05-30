"""Every YAML in ``configs/`` parses and conforms to the basic schema
the training drivers depend on.

This is a structural test, not a semantic one — we do not validate
hyperparameter values, only that the keys the training loops read are
present and of the expected shape.
"""

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def _all_configs() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def test_configs_dir_exists():
    assert CONFIGS_DIR.is_dir()
    assert _all_configs(), "expected at least one YAML in configs/"


@pytest.mark.parametrize("path", _all_configs(), ids=lambda p: p.name)
def test_config_parses(path: Path):
    cfg = yaml.safe_load(path.read_text())
    assert isinstance(cfg, dict), f"{path.name} did not parse as a mapping"


@pytest.mark.parametrize("path", _all_configs(), ids=lambda p: p.name)
def test_config_required_top_level_keys(path: Path):
    cfg = yaml.safe_load(path.read_text())
    for key in ("experiment_name", "seed", "device",
                "data", "model", "optim", "training", "output"):
        assert key in cfg, f"{path.name} missing top-level key {key!r}"
    assert isinstance(cfg["seed"], int)
    assert cfg["device"] in {"auto", "cpu", "cuda"}, \
        f"{path.name} has unsupported device={cfg['device']!r}"


@pytest.mark.parametrize("path", _all_configs(), ids=lambda p: p.name)
def test_config_block_shapes(path: Path):
    cfg = yaml.safe_load(path.read_text())

    data = cfg["data"]
    assert isinstance(data, dict) and "name" in data
    assert "batch_size" in data and isinstance(data["batch_size"], int)
    assert "num_workers" in data and isinstance(data["num_workers"], int)

    model = cfg["model"]
    assert isinstance(model, dict) and "name" in model

    optim_cfg = cfg["optim"]
    assert "name" in optim_cfg and "lr" in optim_cfg
    assert optim_cfg["name"] in {"sgd", "adam", "adamw"}

    training = cfg["training"]
    assert "num_epochs" in training and isinstance(training["num_epochs"], int)
    assert "log_every" in training and isinstance(training["log_every"], int)

    output = cfg["output"]
    assert "checkpoint_dir" in output
    assert "log_dir" in output


def test_experiment_names_unique():
    seen = {}
    for path in _all_configs():
        cfg = yaml.safe_load(path.read_text())
        name = cfg.get("experiment_name")
        if name in seen:
            pytest.fail(
                f"experiment_name {name!r} is duplicated between "
                f"{seen[name]} and {path.name}")
        seen[name] = path.name
