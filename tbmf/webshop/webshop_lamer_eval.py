"""LaMer WebShop evaluator with per-attempt success signals."""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def webshop_lamer_evaluator(task: dict, episode: Episode) -> EvalOutput:
    won = bool(episode.artifacts.get("won", False))
    episodes_played = int(episode.artifacts.get("episodes_played", 0))
    episode_rewards = episode.artifacts.get("episode_rewards", [])
    num_episodes = int(episode.artifacts.get("num_episodes", 3))

    signals = [
        Signal(name="accuracy", value=1.0 if won else 0.0),
        Signal(name="episodes_played", value=float(episodes_played)),
        Signal(name="turns_total", value=float(episode.artifacts.get("turns_total", 0))),
    ]
    for n in range(1, num_episodes + 1):
        solved = 1.0 if any(r > 0 for r in episode_rewards[:n]) else 0.0
        signals.append(Signal(name=f"success_at{n}", value=solved))

    return EvalOutput(reward=1.0 if won else 0.0, is_correct=won, signals=signals)
