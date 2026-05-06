"""Evaluator for the LangGraph math agent.

The flow returns ``None``, so ``episode.artifacts`` is empty. The evaluator
extracts the answer directly from the final assistant message captured by
the gateway — that's the canonical "evaluator parses the Trajectory" pattern.
"""

from __future__ import annotations

import re

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy
from rllm.types import Episode


def _extract_boxed(text: str) -> str | None:
    """Extract the contents of the last ``\\boxed{...}`` with balanced braces."""
    idx = text.rfind(r"\boxed{")
    if idx < 0:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def _extract_answer(text: str) -> str:
    """Try multiple patterns to extract the final answer from assistant text."""
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([+-]?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    numbers = re.findall(r"(?<!\w)([+-]?\d[\d,]*\.?\d*)(?!\w)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def _last_assistant_text(episode: Episode) -> str:
    """Return the last assistant message content from the gateway-captured trajectory.

    Walks back through Steps until it finds one with a non-empty
    ``model_response``. The trailing Step in a tool-using ReAct agent is
    always an assistant message (LangGraph stops once the LLM returns text
    without further tool calls).
    """
    if not episode.trajectories:
        return ""
    for step in reversed(episode.trajectories[-1].steps):
        if step.model_response:
            return step.model_response
    return ""


@rllm.evaluator
def langgraph_math_evaluator(task: dict, episode: Episode) -> EvalOutput:
    """Grade the agent's answer against ground truth using symbolic math comparison."""
    answer_text = _extract_answer(_last_assistant_text(episode))
    ground_truth = str(task.get("answer") or task.get("ground_truth") or "")

    is_correct = grade_answer_mathd(answer_text, ground_truth) or grade_answer_sympy(answer_text, ground_truth)
    reward = 1.0 if is_correct else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
    )
