"""IoU score function: bounding-box Intersection-over-Union."""

from __future__ import annotations

import re

from rllm.eval.score_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.task import Task
from rllm.types import Episode


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    answer_text = extract_answer_text(episode)
    pred_bbox = _parse_bbox(answer_text)
    gt_bbox = task.metadata.get("ground_truth_bbox", task.metadata.get("ground_truth"))

    if pred_bbox is None or gt_bbox is None:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="iou", value=0.0)],
            metadata={"reason": "parse_failure"},
        )

    if isinstance(gt_bbox, str):
        gt_bbox = _parse_bbox(gt_bbox)
    if isinstance(gt_bbox, list | tuple) and len(gt_bbox) == 4:
        gt_bbox = [float(x) for x in gt_bbox]
    else:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="iou", value=0.0)],
            metadata={"reason": "invalid_ground_truth"},
        )

    iou = _compute_iou(pred_bbox, gt_bbox)
    is_correct = iou >= 0.5
    return EvalOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        signals=[Signal(name="iou", value=iou)],
    )


def _parse_bbox(text: str) -> list[float] | None:
    m = re.search(
        r"\[?\s*(\d+(?:\.\d+)?)\s*[,\s]\s*(\d+(?:\.\d+)?)\s*[,\s]\s*(\d+(?:\.\d+)?)\s*[,\s]\s*(\d+(?:\.\d+)?)\s*\]?",
        text,
    )
    return [float(m.group(i)) for i in range(1, 5)] if m else None


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
