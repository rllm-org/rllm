"""Math tool agent evaluator: checks if the final answer matches ground truth."""

from __future__ import annotations

import re

import rllm
from rllm.experimental.eval.types import EvalOutput, Signal, _extract_agent_answer
from rllm.types import Episode


def _normalize_number(text: str) -> float | None:
    """Try to parse a numeric string, stripping commas and whitespace."""
    text = text.strip().replace(",", "")
    # Remove trailing period (e.g. "42.")
    text = text.rstrip(".")
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


@rllm.evaluator
def math_tool_evaluator(task: dict, episode: Episode) -> EvalOutput:
    """Grade the agent's answer by numeric comparison with ground truth."""
    answer_text = _extract_agent_answer(episode)
    ground_truth = str(task.get("ground_truth", ""))

    pred = _normalize_number(answer_text)
    expected = _normalize_number(ground_truth)

    if pred is not None and expected is not None:
        is_correct = abs(pred - expected) < 1e-6
    else:
        is_correct = False

    reward = 1.0 if is_correct else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
    )
