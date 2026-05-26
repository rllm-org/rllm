"""ALFWorld evaluator: reward = 1.0 if the agent completed the task, else 0.0.

The flow records success in ``episode.artifacts["won"]`` and on
``episode.is_correct``; the evaluator translates that into a reward
plus named signals for logging.
"""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def alfworld_evaluator(task: dict, episode: Episode) -> EvalOutput:
    won = bool(episode.artifacts.get("won", False))
    reward = 1.0 if won else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=won,
        signals=[
            Signal(name="accuracy", value=reward),
            Signal(name="turns", value=float(episode.artifacts.get("turns", 0))),
            Signal(name="task_type", value=episode.artifacts.get("task_type", "unknown")),
        ],
    )
