"""Wide-search score function: LLM-judged search results scoring."""

from __future__ import annotations

from rllm.eval.types import EvalOutput
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.eval.evaluator.widesearch import WideSearchEvaluator

    return WideSearchEvaluator().evaluate(task.metadata, episode)
