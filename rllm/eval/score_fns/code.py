"""Code score function: execute generated code against hidden tests.

Wraps :class:`rllm.rewards.code_reward.RewardCodeFn`.
"""

from __future__ import annotations

from rllm.eval.score_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    from rllm.rewards.code_reward import RewardCodeFn
    from rllm.rewards.reward_types import RewardConfig

    answer_text = extract_answer_text(episode)
    reward_fn = RewardCodeFn(RewardConfig())
    reward_output = reward_fn(task_info=task.metadata, action=answer_text)

    is_correct = reward_output.reward > 0
    return EvalOutput(
        reward=float(reward_output.reward),
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=1.0 if is_correct else 0.0)],
        metadata=reward_output.metadata,
    )
