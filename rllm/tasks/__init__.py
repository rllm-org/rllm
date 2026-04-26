"""rLLM task format: load and execute task directories as RL environments.

A task directory contains a ``task.toml`` (Harbor-compatible config),
``instruction.md``, optional ``environment/`` files, and ``tests/``
evaluation scripts.  A dataset is a collection of tasks described by
a ``dataset.toml`` manifest.

Key classes:

- :class:`LoadedTask` / :func:`load_task` — parse a single task directory
- :class:`DatasetConfig` / :func:`load_dataset_config` — parse a dataset manifest
- :class:`BenchmarkLoader` — unified entry point for CLI (local path → Dataset + AgentFlow + Evaluator)
- :class:`TaskExecutor` — sandboxed agent that runs on a task directory
- :class:`TaskEvaluator` — runs ``tests/`` scripts inside the sandbox
- :class:`SimpleEvaluator` — wraps a user's Python evaluate function
"""

from rllm.tasks.dataset_config import DatasetConfig, EvaluationConfig, TaskRef, load_dataset_config
from rllm.tasks.loader import BenchmarkLoader, BenchmarkResult
from rllm.tasks.simple_evaluator import SimpleEvaluator
from rllm.tasks.task_config import LoadedTask, RllmExtensions, load_task
from rllm.tasks.task_evaluator import TaskEvaluator
from rllm.tasks.task_executor import TaskExecutor

__all__ = [
    "BenchmarkLoader",
    "BenchmarkResult",
    "DatasetConfig",
    "EvaluationConfig",
    "LoadedTask",
    "RllmExtensions",
    "SimpleEvaluator",
    "TaskEvaluator",
    "TaskExecutor",
    "TaskRef",
    "load_dataset_config",
    "load_task",
]
