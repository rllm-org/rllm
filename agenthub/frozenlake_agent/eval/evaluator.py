"""FrozenLake evaluator: scores episodes based on goal-reaching success."""

from __future__ import annotations

from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.types import Episode


class FrozenLakeEvaluator:
    """Evaluator that checks whether the agent reached the goal."""

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        success = episode.artifacts.get("success", False)
        num_steps = episode.artifacts.get("num_steps", 0)

        reward = 1.0 if success else 0.0
        return EvalOutput(
            reward=reward,
            is_correct=bool(success),
            signals=[
                Signal("success", float(success)),
                Signal("num_steps", float(num_steps)),
            ],
            metadata={"num_steps": num_steps},
        )
