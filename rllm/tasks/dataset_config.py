"""Dataset configuration: parse dataset.toml manifests.

A dataset.toml describes a collection of tasks — either a flat data file
with a Python evaluator (``type = "simple"``) or a directory of sandbox
task directories (``type = "sandbox"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass
class TaskRef:
    """Pointer to a task directory within a sandbox dataset."""

    path: str


@dataclass
class EvaluationConfig:
    """Evaluation function reference for simple datasets."""

    module: str  # Python file path relative to dataset dir
    function: str = "evaluate"  # Function name in that module


@dataclass
class DatasetConfig:
    """Parsed from ``dataset.toml``."""

    name: str
    type: str = "sandbox"  # "simple" | "sandbox"
    version: str = "1.0"
    description: str = ""

    # Simple datasets only
    data: str = ""  # path to data file relative to dataset dir
    evaluation: EvaluationConfig | None = None

    # Sandbox datasets only
    default_sandbox: str = "docker"

    # Shared
    default_agent: str | None = None
    split: str = "test"

    # Explicit task list (sandbox only, auto-discovered if omitted)
    tasks: list[TaskRef] | None = None


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Parse a ``dataset.toml`` file.

    Args:
        path: Path to the ``dataset.toml`` file.

    Returns:
        A ``DatasetConfig`` instance.
    """
    path = Path(path)
    raw = tomllib.loads(path.read_text())

    ds = raw.get("dataset", {})

    # Parse evaluation section
    evaluation = None
    eval_raw = ds.get("evaluation") or raw.get("evaluation")
    if eval_raw and isinstance(eval_raw, dict):
        evaluation = EvaluationConfig(
            module=eval_raw.get("module", ""),
            function=eval_raw.get("function", "evaluate"),
        )

    # Parse task refs
    tasks = None
    tasks_raw = ds.get("tasks") or raw.get("tasks")
    if tasks_raw and isinstance(tasks_raw, list):
        tasks = [TaskRef(path=t["path"]) if isinstance(t, dict) else TaskRef(path=str(t)) for t in tasks_raw]

    return DatasetConfig(
        name=ds.get("name", path.parent.name),
        type=ds.get("type", "sandbox"),
        version=ds.get("version", "1.0"),
        description=ds.get("description", ""),
        data=ds.get("data", ""),
        evaluation=evaluation,
        default_sandbox=ds.get("default_sandbox", "docker"),
        default_agent=ds.get("default_agent"),
        split=ds.get("split", "test"),
        tasks=tasks,
    )
