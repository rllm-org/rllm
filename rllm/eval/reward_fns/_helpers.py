"""Shared helpers for score_fn modules."""

from __future__ import annotations

from rllm.types import Episode


def extract_answer_text(episode: Episode) -> str:
    """Extract the agent's final answer text from an Episode.

    Fallback chain:

    1. ``episode.artifacts["answer"]`` — cookbook convention (e.g.
       ``cookbooks/math`` sets it explicitly).
    2. ``trajectory.output`` — set by the legacy SDK ``@trajectory``
       decorator and a few harnesses.
    3. ``trajectory.steps[-1].output`` — agent-populated step output
       (legacy harnesses like ``ReActHarness``, ``BashHarness``).
    4. ``trajectory.steps[*].model_response`` walked backwards — gateway-
       captured Steps from ``return None`` flows. The last assistant turn
       in a tool-using ReAct loop has the final text; intermediate
       tool-call steps have empty ``model_response``, so we walk back to
       skip them.
    """
    if "answer" in episode.artifacts:
        return str(episode.artifacts["answer"])
    if not episode.trajectories:
        return ""
    traj = episode.trajectories[-1]
    if traj.output:
        return str(traj.output)
    if not traj.steps:
        return ""
    if traj.steps[-1].output:
        return str(traj.steps[-1].output)
    # Last-resort: walk back through Steps to find the last non-empty
    # `model_response` (the LLM's final assistant message, captured by
    # the gateway). Tool-call turns have empty `model_response` so we
    # skip past them.
    for step in reversed(traj.steps):
        if step.model_response:
            return str(step.model_response)
    return ""
