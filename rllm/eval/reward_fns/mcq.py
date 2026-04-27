"""MCQ score function: extract A–J letter from response, compare to ground truth."""

from __future__ import annotations

import re

from rllm.eval.reward_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task

SYSTEM_PROMPT = "Choose the correct answer from the given options. Think through the problem carefully, then respond with ONLY the letter of the correct answer (A, B, C, D, etc.)."


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    answer_text = extract_answer_text(episode)
    model_letter = _extract_choice_letter(answer_text)

    gt = task.metadata.get("ground_truth", "")
    expected_letter = str(gt).strip().upper()[:1] if gt else ""

    is_correct = model_letter != "" and model_letter == expected_letter
    reward = 1.0 if is_correct else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
        metadata={"model_answer": model_letter, "expected": expected_letter},
    )


def _extract_choice_letter(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper()
    m = re.search(r"(?:answer\s*(?:is|:)\s*\(?([A-Ja-j])\)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:\*\*([A-J])\*\*|\(([A-J])\))", text)
    if m:
        return (m.group(1) or m.group(2)).upper()
    m = re.search(r"\b([A-J])\b", text)
    if m:
        return m.group(1)
    return ""
