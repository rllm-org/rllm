"""Evaluator: check if the bash harness produced the expected output.

BashHarness stores assistant messages in step.output, not command stdout.
For simple "print X" tasks the LLM nearly always mentions the result in its
response, so we scan step outputs for the expected value as a substring.

Known limitation: short expected values (e.g. "8") can false-positive against
unrelated numbers in the response. Acceptable for this e2e demo; production
use should store command stdout in Episode.artifacts (separate PR).
"""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task


@rllm.evaluator
def sandbox_code_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    meta = task.metadata if isinstance(task, Task) else task or {}
    expected = str(meta.get("expected_output", "")).strip()

    if not expected:
        return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no expected_output in task"})

    found_in = ""
    for traj in episode.trajectories:
        for step in traj.steps:
            text = str(step.output or "")
            if expected in text:
                found_in = text[:200]
                break
        if found_in:
            break

    is_correct = bool(found_in)
    return EvalOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=1.0 if is_correct else 0.0)],
        metadata={"expected": expected, "found_in_step": found_in},
    )
