"""BenchmarkLoader: unified entry point for loading local benchmark directories.

Detects whether a CLI ``benchmark`` argument is a local path (vs a catalog name)
and loads it into the standard ``(Dataset, AgentFlow, Evaluator)`` triple that
EvalRunner and the training loop expect.

Supports three cases:

1. Directory with ``dataset.toml`` → parse manifest (simple or sandbox)
2. Directory with ``task.toml`` → single sandbox task
3. Directory with subdirectories containing ``task.toml`` → auto-discover
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rllm.data.dataset import Dataset
from rllm.tasks.dataset_config import DatasetConfig, load_dataset_config
from rllm.tasks.task import Task

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """What the loader returns to the CLI."""

    dataset: Dataset
    agent: Any | None = None  # AgentFlow (TaskRunner with harness, or None for simple datasets)
    evaluator: Any | None = None  # Evaluator (TaskEvaluator or SimpleEvaluator)
    catalog_entry: dict = field(default_factory=dict)


class BenchmarkLoader:
    """Detects and loads local benchmark directories."""

    @staticmethod
    def is_local_benchmark(benchmark: str) -> bool:
        """Check if *benchmark* is a local path containing tasks."""
        if not os.path.exists(benchmark):
            return False
        p = Path(benchmark).resolve()
        if not p.is_dir():
            return False
        if (p / "dataset.toml").exists():
            return True
        if (p / "task.toml").exists():
            return True
        return any((d / "task.toml").exists() for d in p.iterdir() if d.is_dir())

    @staticmethod
    def load(
        benchmark_path: str,
        sandbox_backend: str | None = None,
        harness_name: str | None = None,
    ) -> BenchmarkResult:
        """Load a local benchmark directory.

        Args:
            benchmark_path: Path to the benchmark directory.
            sandbox_backend: Sandbox backend override from ``--sandbox-backend``.
            harness_name: Harness name override from ``--agent`` (e.g.
                ``"react"``, ``"claude-code"``). If ``None``, defaults to
                ``"react"`` for sandbox tasks.
        """
        path = Path(benchmark_path).resolve()

        if (path / "dataset.toml").exists():
            config = load_dataset_config(path / "dataset.toml")
            if config.type == "simple":
                return _load_simple(path, config)
            return _load_sandbox(path, config, sandbox_backend, harness_name)

        if (path / "task.toml").exists():
            return _load_sandbox_single(path, sandbox_backend, harness_name)

        return _load_sandbox_auto(path, sandbox_backend, harness_name)


# ---------------------------------------------------------------------------
# Simple dataset
# ---------------------------------------------------------------------------


def _load_simple(path: Path, config: DatasetConfig) -> BenchmarkResult:
    """Load a simple dataset: data file + Python evaluator."""
    from rllm.tasks.simple_evaluator import SimpleEvaluator

    data_path = path / config.data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    dataset = Dataset.load_data(str(data_path))
    dataset.name = config.name
    dataset.split = config.split

    evaluator = None
    if config.evaluation and config.evaluation.module:
        eval_path = path / config.evaluation.module
        evaluator = SimpleEvaluator.from_file(eval_path, config.evaluation.function)

    return BenchmarkResult(
        dataset=dataset,
        agent=None,
        evaluator=evaluator,
        catalog_entry={
            "description": config.description,
            "category": "custom",
            "default_agent": config.default_agent,
        },
    )


# ---------------------------------------------------------------------------
# Sandbox dataset
# ---------------------------------------------------------------------------


def _load_sandbox(
    path: Path,
    config: DatasetConfig,
    sandbox_backend: str | None,
    harness_name: str | None,
) -> BenchmarkResult:
    """Load a sandbox dataset: directory of task directories."""
    if config.tasks:
        task_dirs = [(path / ref.path).resolve() for ref in config.tasks]
    else:
        task_dirs = _discover_task_dirs(path)

    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {path}")

    dataset = _task_dirs_to_dataset(task_dirs, config.name, config.split)
    backend = sandbox_backend or config.default_sandbox
    agent, evaluator = _make_runner_and_evaluator(backend, harness_name or config.default_agent)

    return BenchmarkResult(
        dataset=dataset,
        agent=agent,
        evaluator=evaluator,
        catalog_entry={
            "description": config.description,
            "category": "agentic",
            "default_agent": harness_name or config.default_agent or "react",
            "reward_fn": "task_script",
        },
    )


def _load_sandbox_single(
    path: Path,
    sandbox_backend: str | None,
    harness_name: str | None,
) -> BenchmarkResult:
    """Wrap a single task directory as a one-element sandbox dataset."""
    loaded = Task.load(path)
    dataset = _task_dirs_to_dataset([path], loaded.name, "test")
    backend = sandbox_backend or loaded.required_sandbox_backend()
    if backend == "any":
        backend = "docker"
    agent, evaluator = _make_runner_and_evaluator(backend, harness_name)

    return BenchmarkResult(
        dataset=dataset,
        agent=agent,
        evaluator=evaluator,
        catalog_entry={
            "description": loaded.config.get("task", {}).get("description", ""),
            "category": "agentic",
            "default_agent": harness_name or "react",
            "reward_fn": "task_script",
        },
    )


def _load_sandbox_auto(
    path: Path,
    sandbox_backend: str | None,
    harness_name: str | None,
) -> BenchmarkResult:
    """Auto-discover subdirectories with task.toml."""
    task_dirs = _discover_task_dirs(path)
    if not task_dirs:
        raise FileNotFoundError(f"No task directories (with task.toml) found in {path}")

    dataset = _task_dirs_to_dataset(task_dirs, path.name, "test")
    backend = sandbox_backend or "docker"
    agent, evaluator = _make_runner_and_evaluator(backend, harness_name)

    return BenchmarkResult(
        dataset=dataset,
        agent=agent,
        evaluator=evaluator,
        catalog_entry={
            "description": f"Auto-discovered tasks from {path.name}",
            "category": "agentic",
            "default_agent": harness_name or "react",
            "reward_fn": "task_script",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_task_dirs(root: Path) -> list[Path]:
    """Find all immediate subdirectories of *root* that contain task.toml."""
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "task.toml").exists())


def _task_dirs_to_dataset(task_dirs: list[Path], name: str, split: str) -> Dataset:
    """Convert task directories into an rLLM Dataset.

    Each row contains ``task_id``, ``task_path``, ``instruction``, ``metadata``.
    """
    rows: list[dict] = []
    for task_dir in task_dirs:
        loaded = Task.load(task_dir)
        rows.append(
            {
                "task_id": loaded.name,
                "task_path": str(loaded.path),
                "instruction": loaded.instruction,
                "metadata": loaded.metadata,
            }
        )
    return Dataset(data=rows, name=name, split=split)


def _make_runner_and_evaluator(backend: str, harness_name: str | None) -> tuple:
    """Build a ``TaskRunner`` (with selected harness) + ``TaskEvaluator``."""
    from rllm.tasks.evaluator import TaskEvaluator
    from rllm.tasks.harness import load_harness
    from rllm.tasks.runner import TaskRunner

    harness = load_harness(harness_name or "react")
    runner = TaskRunner(harness=harness, sandbox_backend=backend)
    evaluator = TaskEvaluator()
    return runner, evaluator
