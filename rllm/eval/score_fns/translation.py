"""Translation score function: chrF / sacreBLEU-style scoring."""

from __future__ import annotations

from rllm.eval.types import EvalOutput
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.eval.evaluator.translation import TranslationEvaluator

    return TranslationEvaluator().evaluate(task.metadata, episode)
