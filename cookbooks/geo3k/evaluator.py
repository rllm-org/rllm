"""Geo3K evaluator: scores geometry answers using math grading."""

from __future__ import annotations

import rllm
from rllm.experimental.eval.types import EvalOutput, Signal, _extract_agent_answer
from rllm.types import Episode


@rllm.evaluator
def geo3k_evaluator(task: dict, episode: Episode) -> EvalOutput:
    """Grade geometry answers by extracting the boxed answer and comparing to ground truth."""
    from rllm.rewards.math_utils.utils import extract_answer, grade_answer_mathd, grade_answer_sympy

    answer_text = _extract_agent_answer(episode)
    model_answer = extract_answer(answer_text)

    if model_answer is None:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
        )

    ground_truth = task.get("ground_truth")
    if ground_truth is None:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
        )

    gt_str = str(ground_truth)
    gt_extracted = extract_answer(gt_str) if "\\boxed" in gt_str else gt_str

    is_correct = grade_answer_mathd(model_answer, gt_extracted) or grade_answer_sympy(model_answer, gt_extracted)
    reward = 1.0 if is_correct else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
    )
