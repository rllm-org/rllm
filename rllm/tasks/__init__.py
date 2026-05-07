"""rLLM tasks: benchmark directory loader.

After PR 2 the heavy lifting moved up to ``rllm.types`` (Task data model)
and ``rllm.engine.agentflow_engine`` (driven by
``rllm.eval._hooks.EvalHooks`` for the eval path). This package keeps
the benchmark-directory loader and the dataset config schema.

Built-in agent flows (``react``, ``bash``, ``claude-code``) are listed in
``rllm/registry/agents.json`` and resolved through
:func:`rllm.eval.agent_loader.load_agent`.
"""

from rllm.tasks.dataset_config import DatasetConfig, EvaluationConfig, TaskRef, load_dataset_config
from rllm.tasks.loader import BenchmarkLoader, BenchmarkResult

__all__ = [
    "BenchmarkLoader",
    "BenchmarkResult",
    "DatasetConfig",
    "EvaluationConfig",
    "TaskRef",
    "load_dataset_config",
]
