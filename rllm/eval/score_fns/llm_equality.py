"""LLM-equality score function: LLM-as-judge semantic equivalence."""

from __future__ import annotations

from rllm.eval.types import EvalOutput
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.eval.evaluator.llm_equality import LLMEqualityEvaluator

    return LLMEqualityEvaluator().evaluate(task.metadata, episode)
