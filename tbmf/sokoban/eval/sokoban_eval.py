"""Sokoban evaluator: reward = 1.0 if all boxes reached targets, else 0.0."""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def sokoban_evaluator(task: dict, episode: Episode) -> EvalOutput:
    won = bool(episode.artifacts.get("won", False))
    reward = 1.0 if won else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=won,
        signals=[
            Signal(name="accuracy", value=reward),
            Signal(name="turns", value=float(episode.artifacts.get("turns", 0))),
            Signal(name="env_steps", value=float(episode.artifacts.get("env_steps", 0))),
            Signal(name="num_boxes", value=float(episode.artifacts.get("num_boxes", 0))),
        ],
    )
