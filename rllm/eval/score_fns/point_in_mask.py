"""Point-in-mask score function: spatial reasoning verifier."""

from __future__ import annotations

import re

from rllm.eval.score_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    answer_text = extract_answer_text(episode)
    point = _parse_point(answer_text)

    if point is None:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="point_accuracy", value=0.0)],
            metadata={"reason": "parse_failure"},
        )

    mask_data = task.metadata.get("ground_truth_mask")
    if mask_data is None:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="point_accuracy", value=0.0)],
            metadata={"reason": "no_mask"},
        )

    try:
        is_in_mask = _check_point_in_mask(point, mask_data)
    except Exception:
        is_in_mask = False

    reward = 1.0 if is_in_mask else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=is_in_mask,
        signals=[Signal(name="point_accuracy", value=reward)],
    )


def _parse_point(text: str) -> tuple[float, float] | None:
    m = re.search(r"\(?\s*(\d+(?:\.\d+)?)\s*[,\s]\s*(\d+(?:\.\d+)?)\s*\)?", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _check_point_in_mask(point: tuple[float, float], mask_data) -> bool:
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(mask_data)) if isinstance(mask_data, bytes) else mask_data
    img = img.convert("L")
    x, y = point
    w, h = img.size
    px = int(x * w / 1000) if x > 1 else int(x * w)
    py = int(y * h / 1000) if y > 1 else int(y * h)
    px = max(0, min(px, w - 1))
    py = max(0, min(py, h - 1))
    return img.getpixel((px, py)) > 127
