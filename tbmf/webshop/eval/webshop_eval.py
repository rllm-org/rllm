"""WebShop evaluator: reward = 1.0 if the purchase succeeded, else 0.0."""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def webshop_evaluator(task: dict, episode: Episode) -> EvalOutput:
    won = bool(episode.artifacts.get("won", False))
    task_score = float(episode.artifacts.get("task_score", 0.0))
    reward = 1.0 if won else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=won,
        signals=[
            Signal(name="accuracy", value=reward),
            Signal(name="task_score", value=task_score),
            Signal(name="turns", value=float(episode.artifacts.get("turns", 0))),
            Signal(name="env_steps", value=float(episode.artifacts.get("env_steps", 0))),
        ],
        metadata={
            "won": won,
            "task_score": task_score,
            "session_id": episode.artifacts.get("session_id"),
        },
    )
