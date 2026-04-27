"""BFCL score function: AST-based function-call verification.

Thin wrapper over :class:`rllm.eval.evaluator.bfcl.BFCLEvaluator` that
adapts the new ``(task: Task, episode)`` signature to the legacy
``(task: dict, episode)`` form.
"""

from __future__ import annotations

from rllm.eval.types import EvalOutput
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.eval.evaluator.bfcl import BFCLEvaluator

    return BFCLEvaluator().evaluate(task.metadata, episode)
