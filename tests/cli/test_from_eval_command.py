"""Tests for `rllm dataset from-eval`."""

import json
import os

import pytest
from click.testing import CliRunner

from rllm.cli.main import cli
from rllm.eval.results import EvalItem, EvalResult


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    from rllm.data.dataset import DatasetRegistry

    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", rllm_home)
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", os.path.join(rllm_home, "datasets", "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", os.path.join(rllm_home, "datasets"))
    legacy_dir = str(tmp_path / "legacy_registry")
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_REGISTRY_DIR", legacy_dir)
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_REGISTRY_FILE", os.path.join(legacy_dir, "dataset_registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_DATASET_DIR", os.path.join(legacy_dir, "datasets"))
    return rllm_home


@pytest.fixture
def runner():
    return CliRunner()


def _episode(user: str, assistant: str) -> dict:
    return {"trajectories": [{"name": "default", "steps": [{"chat_completions": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}]}]}


def _make_run(run_dir, *, attempts, rollouts):
    """rollouts: list of (task_idx, attempt, is_correct, assistant_text)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes").mkdir()
    items = []
    for task_idx, attempt, is_correct, assistant in rollouts:
        reward = 1.0 if is_correct else 0.0
        items.append(EvalItem(idx=task_idx, attempt=attempt, reward=reward, is_correct=is_correct, signals={"accuracy": reward}))
        eval_idx = task_idx * attempts + attempt
        (run_dir / "episodes" / f"episode_{eval_idx:06d}_t{task_idx}.json").write_text(json.dumps(_episode(f"q{task_idx}", assistant)))
    EvalResult.from_items("bench", "model", "agent", items, attempts=attempts).save(str(run_dir / "results.json"))
    return str(run_dir)


@pytest.fixture
def run_path(tmp_path):
    return _make_run(
        tmp_path / "run1",
        attempts=2,
        rollouts=[
            (0, 0, False, "wrong a"),
            (0, 1, False, "wrong b"),
            (1, 0, True, "t1 correct"),
            (1, 1, False, "t1 wrong"),
            (2, 0, True, "t2 correct"),
            (2, 1, True, "t2 also correct"),
        ],
    )


def test_dry_run_registers_nothing(runner, tmp_rllm_home, run_path):
    from rllm.data import DatasetRegistry

    result = runner.invoke(cli, ["dataset", "from-eval", run_path, "--name", "curated", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert DatasetRegistry.load_dataset("curated", "train") is None


def test_register_default(runner, tmp_rllm_home, run_path):
    from rllm.data import DatasetRegistry

    result = runner.invoke(cli, ["dataset", "from-eval", run_path, "--name", "curated"])
    assert result.exit_code == 0, result.output
    assert "Registered" in result.output
    ds = DatasetRegistry.load_dataset("curated", "train")
    assert ds is not None
    # t1 (1 correct) + t2 (2 correct) = 3 rows
    assert len(ds) == 3
    assert all("messages" in row for row in ds.data)


def test_difficulty_band_filter(runner, tmp_rllm_home, run_path):
    from rllm.data import DatasetRegistry

    result = runner.invoke(cli, ["dataset", "from-eval", run_path, "--name", "band", "--filter", "0 < avg < 1"])
    assert result.exit_code == 0, result.output
    ds = DatasetRegistry.load_dataset("band", "train")
    assert len(ds) == 1  # only t1


def test_val_fraction_registers_two_splits(runner, tmp_rllm_home, run_path):
    from rllm.data import DatasetRegistry

    result = runner.invoke(cli, ["dataset", "from-eval", run_path, "--name", "withval", "--val-fraction", "0.5", "--seed", "0"])
    assert result.exit_code == 0, result.output
    splits = set(DatasetRegistry.get_dataset_splits("withval"))
    assert {"train", "test"} <= splits


def test_no_match_exits_nonzero(runner, tmp_rllm_home, run_path):
    result = runner.invoke(cli, ["dataset", "from-eval", run_path, "--name", "none", "--filter", "n_correct >= 99"])
    assert result.exit_code == 1
    assert "No trajectories matched" in result.output


def test_bad_filter_exits_nonzero(runner, tmp_rllm_home, run_path):
    result = runner.invoke(cli, ["dataset", "from-eval", run_path, "--name", "bad", "--filter", "__import__('os')"])
    assert result.exit_code == 1


def test_best_n_requires_max_per_task(runner, tmp_rllm_home, run_path):
    result = runner.invoke(cli, ["dataset", "from-eval", run_path, "--name", "bn", "--select", "best-n"])
    assert result.exit_code == 1
    assert "max-per-task" in result.output


def test_unknown_run_exits_nonzero(runner, tmp_rllm_home, tmp_path):
    result = runner.invoke(cli, ["dataset", "from-eval", str(tmp_path / "nope"), "--name", "x"])
    assert result.exit_code == 1
