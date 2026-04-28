"""Depth score function: absolute relative error on depth estimation."""

from __future__ import annotations

import re

from rllm.eval.reward_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    answer_text = extract_answer_text(episode)
    pred_depth = _parse_depth(answer_text)

    gt_depth_raw = task.metadata.get("ground_truth", "")
    try:
        gt_depth = float(gt_depth_raw)
    except (TypeError, ValueError):
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="absrel", value=1.0)],
            metadata={"reason": "invalid_ground_truth"},
        )

    if pred_depth is None or gt_depth <= 0:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="absrel", value=1.0)],
            metadata={"reason": "parse_failure"},
        )

    absrel = abs(pred_depth - gt_depth) / gt_depth
    reward = max(0.0, 1.0 - absrel)
    is_correct = reward > 0.5
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="absrel", value=absrel)],
    )


def _parse_depth(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meters|metre)?", text)
    return float(m.group(1)) if m else None
