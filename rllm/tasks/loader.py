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
from rllm.tasks.task_config import load_task

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """What the loader returns to the CLI."""

    dataset: Dataset
    agent: Any | None = None  # AgentFlow (optional — CLI --agent can override)
    evaluator: Any | None = None  # Evaluator (optional — CLI --evaluator can override)
    catalog_entry: dict = field(default_factory=dict)


class BenchmarkLoader:
    """Detects and loads local benchmark directories."""

    @staticmethod
    def is_local_benchmark(benchmark: str) -> bool:
        """Check if *benchmark* is a local path containing tasks.

        Returns True if the path exists and looks like a benchmark directory
        (has ``dataset.toml``, ``task.toml``, or subdirs with ``task.toml``).
        """
        if not os.path.exists(benchmark):
            return False
        p = Path(benchmark).resolve()
        if not p.is_dir():
            return False
        if (p / "dataset.toml").exists():
            return True
        if (p / "task.toml").exists():
            return True
        # Check for any subdirectory with task.toml
        return any((d / "task.toml").exists() for d in p.iterdir() if d.is_dir())

    @staticmethod
    def load(benchmark_path: str, sandbox_backend: str | None = None) -> BenchmarkResult:
        """Load a local benchmark directory.

        Args:
            benchmark_path: Path to the benchmark directory.
            sandbox_backend: Sandbox backend override from ``--sandbox-backend`` CLI flag.

        Returns:
            A ``BenchmarkResult`` with dataset, optional agent, optional evaluator,
            and a synthesized catalog entry.
        """
        path = Path(benchmark_path).resolve()

        if (path / "dataset.toml").exists():
            config = load_dataset_config(path / "dataset.toml")
            if config.type == "simple":
                return _load_simple(path, config)
            else:
                return _load_sandbox(path, config, sandbox_backend)

        elif (path / "task.toml").exists():
            # Single task directory
            return _load_sandbox_single(path, sandbox_backend)

        else:
            # Auto-discover subdirectories with task.toml
            return _load_sandbox_auto(path, sandbox_backend)


# ---------------------------------------------------------------------------
# Simple dataset
# ---------------------------------------------------------------------------


def _load_simple(path: Path, config: DatasetConfig) -> BenchmarkResult:
    """Load a simple dataset: data file + Python evaluator."""
    from rllm.tasks.simple_evaluator import SimpleEvaluator

    # Load data
    data_path = path / config.data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    dataset = Dataset.load_data(str(data_path))
    dataset.name = config.name
    dataset.split = config.split

    # Load evaluator
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
) -> BenchmarkResult:
    """Load a sandbox dataset: directory of task directories."""
    # Discover task directories
    if config.tasks:
        task_dirs = [(path / ref.path).resolve() for ref in config.tasks]
    else:
        task_dirs = _discover_task_dirs(path)

    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {path}")

    dataset = _task_dirs_to_dataset(task_dirs, config.name, config.split)
    backend = sandbox_backend or config.default_sandbox
    agent, evaluator = _make_sandbox_agent_evaluator(backend)

    return BenchmarkResult(
        dataset=dataset,
        agent=agent,
        evaluator=evaluator,
        catalog_entry={
            "description": config.description,
            "category": "agentic",
            "default_agent": config.default_agent or "task-executor",
            "reward_fn": "task_script",
        },
    )


def _load_sandbox_single(path: Path, sandbox_backend: str | None) -> BenchmarkResult:
    """Wrap a single task directory as a one-element sandbox dataset."""
    loaded = load_task(path)
    dataset = _task_dirs_to_dataset([path], loaded.task_name, "test")
    backend = sandbox_backend or loaded.rllm.sandbox
    if backend == "any":
        backend = "docker"
    agent, evaluator = _make_sandbox_agent_evaluator(backend)

    return BenchmarkResult(
        dataset=dataset,
        agent=agent,
        evaluator=evaluator,
        catalog_entry={
            "description": loaded.raw_config.get("task", {}).get("description", ""),
            "category": "agentic",
            "default_agent": "task-executor",
            "reward_fn": "task_script",
        },
    )


def _load_sandbox_auto(path: Path, sandbox_backend: str | None) -> BenchmarkResult:
    """Auto-discover subdirectories with task.toml."""
    task_dirs = _discover_task_dirs(path)
    if not task_dirs:
        raise FileNotFoundError(f"No task directories (with task.toml) found in {path}")

    dataset = _task_dirs_to_dataset(task_dirs, path.name, "test")
    backend = sandbox_backend or "docker"
    agent, evaluator = _make_sandbox_agent_evaluator(backend)

    return BenchmarkResult(
        dataset=dataset,
        agent=agent,
        evaluator=evaluator,
        catalog_entry={
            "description": f"Auto-discovered tasks from {path.name}",
            "category": "agentic",
            "default_agent": "task-executor",
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
    """Convert a list of task directories into an rLLM Dataset.

    Each row contains ``task_id``, ``task_path``, ``instruction``, and ``metadata``.
    """
    rows: list[dict] = []
    for task_dir in task_dirs:
        loaded = load_task(task_dir)
        rows.append(
            {
                "task_id": loaded.task_name,
                "task_path": str(loaded.path),
                "instruction": loaded.instruction,
                "metadata": loaded.metadata,
            }
        )
    return Dataset(data=rows, name=name, split=split)


def _make_sandbox_agent_evaluator(backend: str) -> tuple:
    """Create a TaskExecutor + TaskEvaluator pair."""
    from rllm.tasks.task_evaluator import TaskEvaluator
    from rllm.tasks.task_executor import TaskExecutor

    agent = TaskExecutor(sandbox_backend=backend)
    evaluator = TaskEvaluator()
    return agent, evaluator
