"""Shared helpers for score_fn modules."""

from __future__ import annotations

from rllm.types import Episode


def extract_answer_text(episode: Episode) -> str:
    """Extract the agent's final answer text from an Episode.

    Convention: the agent's final answer lives at ``trajectory.output``.
    Falls back to ``episode.artifacts["answer"]`` for legacy AgentFlows
    that stash the answer there.
    """
    if "answer" in episode.artifacts:
        return str(episode.artifacts["answer"])
    if episode.trajectories:
        traj = episode.trajectories[-1]
        if traj.output:
            return str(traj.output)
        if traj.steps:
            return str(traj.steps[-1].output or "")
    return ""
