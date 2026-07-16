"""Evaluator for codex_code: check if the Codex CLI agent produced the expected output.

Codex CLI makes multiple LLM calls per episode (planning, tool use, summarizing).
Each call becomes a Step with `model_response` containing the LLM's text output.
Since the gateway captures traces (not Codex CLI stdout), we check model responses
for the expected answer.

Two matching strategies:
1. Exact line match: expected output appears as a complete line in any step's response
2. Substring match: expected output appears anywhere in the response (fallback)

This is more robust than sandbox_code_evaluator which checks step.output (always None
for CLI harnesses — output is not populated from traces).
"""

from __future__ import annotations

import re

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task


def _normalize(text: str) -> str:
    """Normalize whitespace for comparison."""
    return " ".join(text.split())


def _extract_code_output(text: str) -> list[str]:
    """Extract content from code blocks that look like program output."""
    outputs = []
    in_block = False
    block_lines = []
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if in_block:
                outputs.append("\n".join(block_lines))
                block_lines = []
            in_block = not in_block
            continue
        if in_block:
            block_lines.append(line)
    return outputs


@rllm.evaluator
def codex_code_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    meta = task.metadata if isinstance(task, Task) else task or {}
    expected = str(meta.get("expected_output", "")).strip()

    if not expected:
        return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no expected_output in task"})

    expected_normalized = _normalize(expected)
    best_match = ""
    match_type = ""

    for traj in episode.trajectories:
        for step in traj.steps:
            text = str(step.model_response or "")
            if not text:
                continue

            for line in text.split("\n"):
                if _normalize(line.strip()) == expected_normalized:
                    best_match = line.strip()
                    match_type = "exact_line"
                    break

            if not best_match:
                for block in _extract_code_output(text):
                    if _normalize(block.strip()) == expected_normalized:
                        best_match = block.strip()
                        match_type = "code_block"
                        break

            if not best_match and expected in text:
                idx = text.index(expected)
                best_match = text[max(0, idx - 20):idx + len(expected) + 20]
                match_type = "substring"

            if best_match:
                break
        if best_match:
            break

    is_correct = bool(best_match)
    reward = 1.0 if is_correct else 0.0

    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
        metadata={
            "expected": expected,
            "match_type": match_type,
            "matched_text": best_match[:200] if best_match else "",
        },
    )
