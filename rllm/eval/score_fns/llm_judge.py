"""LLM-judge score function: rubric-based grading by an LLM."""

from __future__ import annotations

from rllm.eval.types import EvalOutput
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.eval.evaluator.llm_judge import LLMJudgeEvaluator

    return LLMJudgeEvaluator().evaluate(task.metadata, episode)
