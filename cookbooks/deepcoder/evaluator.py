"""Deepcoder evaluator: re-run the model's final code against the hidden tests.

Uses :class:`rllm.rewards.code_reward.RewardCodeFn` — the same grader
the flow uses turn-by-turn — so the in-loop signal and the final score
are identical.

The flow stores the model's final code in ``episode.artifacts["answer"]``.
The evaluator pulls it from there and re-grades against the hidden
tests in ``task.metadata`` (Runner path) or ``task`` itself (training
path, where the raw row dict is passed through).
"""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task


def _task_info(task: Task | dict) -> dict:
    """Normalize the ``task`` argument to the dict ``RewardCodeFn`` expects.

    The eval Runner passes a :class:`Task` (use ``.metadata``); the
    training engine passes the raw row dict directly.
    """
    if isinstance(task, Task):
        return task.metadata or {}
    return task or {}


@rllm.evaluator
def deepcoder_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    from rllm.rewards.code_reward import RewardCodeFn
    from rllm.rewards.reward_types import RewardConfig

    answer = str(episode.artifacts.get("answer", ""))
    grader = RewardCodeFn(RewardConfig())
    out = grader(task_info=_task_info(task), action=answer)

    is_correct = bool(out.is_correct)
    return EvalOutput(
        reward=float(out.reward),
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=1.0 if is_correct else 0.0)],
        metadata=out.metadata,
    )
