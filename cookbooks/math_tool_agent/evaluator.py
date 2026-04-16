"""Math tool agent evaluator: checks if the final answer matches ground truth.

Uses rllm's math grading utilities for robust comparison that handles LaTeX,
symbolic expressions, and numeric formats (e.g. ``\\frac{1}{2}`` vs ``0.5``).
"""

from __future__ import annotations

import rllm
from rllm.experimental.eval.types import EvalOutput, Signal, _extract_agent_answer
from rllm.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy
from rllm.types import Episode


@rllm.evaluator
def math_tool_evaluator(task: dict, episode: Episode) -> EvalOutput:
    """Grade the agent's answer against ground truth using symbolic math comparison."""
    answer_text = _extract_agent_answer(episode)
    ground_truth = str(task.get("ground_truth", ""))

    is_correct = grade_answer_mathd(answer_text, ground_truth) or grade_answer_sympy(answer_text, ground_truth)

    reward = 1.0 if is_correct else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
    )
