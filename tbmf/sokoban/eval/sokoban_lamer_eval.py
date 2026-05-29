"""LaMer Sokoban evaluator.

Trajectory rewards are pre-computed in the flow with cross-episode
discounting. This evaluator provides metrics/signals and a fallback
reward for any trajectory that somehow missed its reward assignment.

is_correct is based on FIRST episode success only, so the trainer's pass@1
and pass@4 metrics are directly comparable to single-episode GRPO evaluation.
Cross-episode improvement is captured via success_atN and accuracy_any signals.

Emits per-attempt success signals for computing sequential pass rates:
  - success_at1: solved on first attempt
  - success_at2: solved within first 2 attempts
  - success_at3: solved within first 3 attempts
"""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def sokoban_lamer_evaluator(task: dict, episode: Episode) -> EvalOutput:
    episodes_played = int(episode.artifacts.get("episodes_played", 0))
    episode_rewards = episode.artifacts.get("episode_rewards", [])
    num_episodes = int(episode.artifacts.get("num_episodes", 3))

    first_ep_won = bool(episode_rewards[0] > 0) if episode_rewards else False
    won_any = any(r > 0 for r in episode_rewards)

    signals = [
        Signal(name="accuracy", value=1.0 if first_ep_won else 0.0),
        Signal(name="accuracy_any", value=1.0 if won_any else 0.0),
        Signal(name="episodes_played", value=float(episodes_played)),
        Signal(name="turns_total", value=float(episode.artifacts.get("turns_total", 0))),
        Signal(name="env_steps_total", value=float(episode.artifacts.get("env_steps_total", 0))),
        Signal(name="num_boxes", value=float(episode.artifacts.get("num_boxes", 0))),
    ]
    for n in range(1, num_episodes + 1):
        solved = 1.0 if any(r > 0 for r in episode_rewards[:n]) else 0.0
        signals.append(Signal(name=f"success_at{n}", value=solved))

    return EvalOutput(
        reward=1.0 if first_ep_won else 0.0,
        is_correct=first_ep_won,
        signals=signals,
    )
