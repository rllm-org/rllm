"""Task: a single problem instance.

A Task is **pure data** — no methods, no callbacks. It describes itself
(instruction, metadata) and points at the directory where its verifier
lives. The framework's :class:`rllm.runner.Runner` reads the task config
and resolves the appropriate :class:`rllm.eval.types.Evaluator` at run time.

Two physical shapes both produce ``Task`` instances:

1. **Task-per-directory** (Harbor-style): each ``task-NNN/`` is one Task
   with ``sub_dir`` set to its subdirectory. Verifier lives in
   ``benchmark_dir/sub_dir/tests/``.

2. **Rows-with-shared-verifier** (gsm8k-style): one row from a JSONL file
   becomes one Task. ``sub_dir`` is ``None``; the verifier is shared
   across all rows and lives in ``benchmark_dir/tests/`` (or is referenced
   by name in ``dataset.toml``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Task:
    """A single problem instance.

    All fields are inert data. The framework reads ``benchmark_dir`` and
    ``sub_dir`` to find verifier scripts / config; the rest is content
    the agent and verifier consume.
    """

    id: str
    """Stable identifier (e.g. row index, task-NNN, or harbor task name)."""

    instruction: str | list[dict]
    """What the agent sees. Plain text, or multimodal content blocks."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary task data (ground truth, MCQ choices, harbor task.toml, ...).
    For data tasks, this is the source row. For sandbox tasks, this is the
    parsed ``task.toml`` plus anything else the verifier needs."""

    benchmark_dir: Path = field(default_factory=Path)
    """Path to the benchmark directory (where ``dataset.toml`` lives)."""

    sub_dir: Path | None = None
    """For task-per-directory shape: relative path of this task's subdir
    within ``benchmark_dir``. ``None`` for rows-with-shared-verifier."""

    @property
    def task_dir(self) -> Path:
        """The directory holding *this* task's files.

        For per-task-dir tasks: ``benchmark_dir / sub_dir``.
        For shared-verifier tasks: ``benchmark_dir`` (verifier is shared).
        """
        return self.benchmark_dir / self.sub_dir if self.sub_dir else self.benchmark_dir
