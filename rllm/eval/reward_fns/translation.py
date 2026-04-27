"""Translation score function: ChrF (character n-gram F-score).

ChrF is language-agnostic — works without tokenization. Standard WMT
metric alongside BLEU.
"""

from __future__ import annotations

from collections import Counter

from rllm.eval.reward_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.task import Task
from rllm.types import Episode

SYSTEM_PROMPT = "Translate the given text accurately. Provide only the translation, no explanation or commentary."

_MAX_N = 6
_BETA = 2.0


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    hypothesis = extract_answer_text(episode)
    reference = str(task.metadata.get("ground_truth", ""))

    if not reference:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="chrf", value=0.0)],
            metadata={"reason": "no_reference"},
        )

    score = _compute_chrf(hypothesis, reference)
    is_correct = score >= 0.5
    return EvalOutput(
        reward=score,
        is_correct=is_correct,
        signals=[Signal(name="chrf", value=score)],
    )


def _char_ngrams(text: str, n: int) -> Counter:
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _compute_chrf(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    count = 0
    for n in range(1, _MAX_N + 1):
        hyp = _char_ngrams(hypothesis, n)
        ref = _char_ngrams(reference, n)
        if not hyp or not ref:
            continue
        common = sum((hyp & ref).values())
        hyp_total = sum(hyp.values())
        ref_total = sum(ref.values())
        precision = common / hyp_total if hyp_total > 0 else 0.0
        recall = common / ref_total if ref_total > 0 else 0.0
        total_precision += precision
        total_recall += recall
        count += 1

    if count == 0:
        return 0.0
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    if avg_precision + avg_recall == 0:
        return 0.0
    beta_sq = _BETA**2
    return (1 + beta_sq) * avg_precision * avg_recall / (beta_sq * avg_precision + avg_recall)
