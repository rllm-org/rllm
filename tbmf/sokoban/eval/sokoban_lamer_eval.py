"""LaMer Sokoban evaluator.

Trajectory rewards are pre-computed in the flow with cross-episode
discounting. This evaluator provides metrics/signals and a fallback
reward for any trajectory that somehow missed its reward assignment.

Emits per-attempt success signals for computing sequential pass rates:
  - success_at1: solved on first attempt
  - success_at2: solved within first 2 attempts
  - success_at3: solved within first 3 attempts

When aggregated across rollouts:
  - mean(success_atN) = pass@1atN (single-sample pass rate by attempt N)
  - max across val.rollout.n samples = pass@katN (best-of-k pass rate)
"""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def sokoban_lamer_evaluator(task: dict, episode: Episode) -> EvalOutput:
    won = bool(episode.artifacts.get("won", False))
    episodes_played = int(episode.artifacts.get("episodes_played", 0))
    episode_rewards = episode.artifacts.get("episode_rewards", [])
    num_episodes = int(episode.artifacts.get("num_episodes", 3))
    fallback_reward = 1.0 if won else 0.0

    # Per-attempt cumulative success: success_atN = solved within first N attempts
    success_at = []
    for n in range(1, num_episodes + 1):
        solved = 1.0 if any(r > 0 for r in episode_rewards[:n]) else 0.0
        success_at.append(solved)

    signals = [
        Signal(name="accuracy", value=1.0 if won else 0.0),
        Signal(name="episodes_played", value=float(episodes_played)),
        Signal(name="turns_total", value=float(episode.artifacts.get("turns_total", 0))),
        Signal(name="env_steps_total", value=float(episode.artifacts.get("env_steps_total", 0))),
        Signal(name="num_boxes", value=float(episode.artifacts.get("num_boxes", 0))),
    ]
    for n, val in enumerate(success_at, start=1):
        signals.append(Signal(name=f"success_at{n}", value=val))

    return EvalOutput(
        reward=fallback_reward,
        is_correct=won,
        signals=signals,
    )
