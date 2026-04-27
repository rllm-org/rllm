"""Math score function: \\boxed{} answer extraction + symbolic grading.

Wraps :mod:`rllm.rewards.math_utils` for use as a verifier.
"""

from __future__ import annotations

from rllm.eval.reward_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task

SYSTEM_PROMPT = "Solve the math problem step by step, showing your reasoning clearly. Put your final answer in \\boxed{} notation.\nExample: The answer is \\boxed{42}."


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.rewards.math_utils.utils import extract_answer, grade_answer_mathd, grade_answer_sympy

    answer_text = extract_answer_text(episode)

    model_answer = extract_answer(answer_text)
    if model_answer is None:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"reason": "no_answer_extracted"},
        )

    ground_truths = task.metadata.get("ground_truth")
    if ground_truths is None:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"reason": "no_ground_truth"},
        )

    if isinstance(ground_truths, str | float | int):
        ground_truths = [ground_truths]

    # Process ground truths (extract from boxed if present)
    processed: list[str] = []
    for truth in ground_truths:
        truth_str = str(truth)
        if "\\boxed" in truth_str:
            extracted = extract_answer(truth_str)
            if extracted is not None:
                processed.append(extracted)
        else:
            processed.append(truth_str)

    if not processed:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"reason": "no_processed_ground_truth"},
        )

    for truth in processed:
        if grade_answer_mathd(model_answer, truth) or grade_answer_sympy(model_answer, truth):
            return EvalOutput(
                reward=1.0,
                is_correct=True,
                signals=[Signal(name="accuracy", value=1.0)],
            )

    return EvalOutput(
        reward=0.0,
        is_correct=False,
        signals=[Signal(name="accuracy", value=0.0)],
    )
