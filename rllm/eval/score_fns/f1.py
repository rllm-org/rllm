"""F1 score function: token-overlap F1 between prediction and gold answer."""

from __future__ import annotations

import re
import string
from collections import Counter

from rllm.eval.score_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.task import Task
from rllm.types import Episode

SYSTEM_PROMPT = "Answer the question directly and concisely. Provide only the answer, no additional explanation."


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    answer_text = extract_answer_text(episode)
    gold_text = task.metadata.get("ground_truth", "") or ""

    pred_tokens = _normalize(str(answer_text)).split()
    gold_tokens = _normalize(str(gold_text)).split()

    if not pred_tokens or not gold_tokens:
        f1 = 0.0
    else:
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    is_correct = f1 > 0
    return EvalOutput(
        reward=f1,
        is_correct=is_correct,
        signals=[Signal(name="f1", value=f1)],
    )


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
