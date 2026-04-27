"""IFEval score function: instruction-following constraint checking."""

from __future__ import annotations

from rllm.eval.types import EvalOutput
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.eval.evaluator.ifeval import IFEvalEvaluator

    return IFEvalEvaluator().evaluate(task.metadata, episode)
