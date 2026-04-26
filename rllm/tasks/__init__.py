"""rLLM task system: Task + AgentHarness + TaskRunner.

A **Task** is a directory containing ``task.toml``, ``instruction.md``,
``environment/`` files/Dockerfile, and ``tests/`` evaluation scripts.
A task knows how to materialize itself in a sandbox and how to score
an attempt — but knows nothing about which agent will solve it.

An **AgentHarness** is a pluggable agent driver — it runs an agent
against a task in a sandbox. Built-in harnesses:

- ``react`` — host-side ReAct loop (default)
- ``claude-code`` — Claude Code CLI installed inside the sandbox
- ``codex`` / ``opencode`` — other in-sandbox CLI agents (extensible)

A **TaskRunner** orchestrates everything: it implements ``AgentFlow`` so
it plugs into ``EvalRunner`` and the training engine. Same task, swap the
``--agent`` flag, run a different harness.

A **dataset** is a collection of tasks described by a ``dataset.toml``
manifest (or a directory auto-discovered by :class:`BenchmarkLoader`).

Top-level API:

- :class:`Task` / :func:`Task.load` — load one task directory
- :class:`AgentHarness` (Protocol) and :func:`load_harness`
- :class:`TaskRunner` — orchestrator (implements ``AgentFlow``)
- :class:`TaskEvaluator` — implements ``Evaluator``
- :class:`SimpleEvaluator` — wrap a Python evaluate function
- :class:`BenchmarkLoader` — CLI entry point
- :class:`DatasetConfig` — parsed ``dataset.toml``
"""

from rllm.tasks.dataset_config import DatasetConfig, EvaluationConfig, TaskRef, load_dataset_config
from rllm.tasks.evaluator import TaskEvaluator
from rllm.tasks.harness import AgentHarness, list_harnesses, load_harness, register_harness
from rllm.tasks.loader import BenchmarkLoader, BenchmarkResult
from rllm.tasks.runner import TaskRunner
from rllm.tasks.simple_evaluator import SimpleEvaluator
from rllm.tasks.task import RllmExtensions, Task

__all__ = [
    "AgentHarness",
    "BenchmarkLoader",
    "BenchmarkResult",
    "DatasetConfig",
    "EvaluationConfig",
    "RllmExtensions",
    "SimpleEvaluator",
    "Task",
    "TaskEvaluator",
    "TaskRef",
    "TaskRunner",
    "list_harnesses",
    "load_dataset_config",
    "load_harness",
    "register_harness",
]
