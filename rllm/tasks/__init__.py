"""rLLM tasks: benchmark loader + harness registry.

After PR 2, the heavy lifting moved up to ``rllm.task`` (Task data model)
and ``rllm.runner`` (Runner orchestrator). This package keeps the
benchmark-directory loader and the registry of built-in agent harnesses.
"""

from rllm.tasks.dataset_config import DatasetConfig, EvaluationConfig, TaskRef, load_dataset_config
from rllm.tasks.harness import list_harnesses, load_harness, register_harness
from rllm.tasks.loader import BenchmarkLoader, BenchmarkResult

__all__ = [
    "BenchmarkLoader",
    "BenchmarkResult",
    "DatasetConfig",
    "EvaluationConfig",
    "TaskRef",
    "list_harnesses",
    "load_dataset_config",
    "load_harness",
    "register_harness",
]
