"""Countdown score function: arithmetic puzzle scoring."""

from __future__ import annotations

from rllm.eval.score_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.task import Task
from rllm.types import Episode

SYSTEM_PROMPT = (
    "You are given a target number and a set of numbers. Use each number exactly once "
    "with basic arithmetic (+, -, *, /) to reach the target. Show your reasoning, "
    "then provide your equation inside <answer>...</answer> tags.\n"
    "Example: <answer>(25 + 3) * 2</answer>"
)


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.rewards.countdown_reward import compute_score

    answer_text = extract_answer_text(episode)
    target = task.metadata.get("target")
    nums = task.metadata.get("nums", [])

    if target is None or not nums:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"reason": "missing_target_or_nums"},
        )

    score = compute_score(answer_text, {"target": target, "numbers": nums})
    is_correct = score >= 1.0
    reward = 1.0 if is_correct else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=float(is_correct))],
    )
