"""LLM-based semantic equality score function.

Pipeline: exact normalised match → LLM judge → token F1 fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
from collections import Counter

from rllm.eval.reward_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.task import Task
from rllm.types import Episode

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Answer the question directly and concisely. Your answer will be compared to the ground truth for semantic equivalence."

_EQUALITY_SYSTEM_PROMPT = """\
You are an impartial judge. Determine whether the candidate answer is semantically \
equivalent to the reference answer. Consider meaning, not exact wording. \
Numbers must match in value. Respond with JSON: {"equivalent": true} or {"equivalent": false}."""

_EQUALITY_USER_TEMPLATE = """\
Reference answer: {reference}

Candidate answer: {candidate}

Is the candidate answer semantically equivalent to the reference answer? \
Respond with JSON only."""


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    answer_text = extract_answer_text(episode)
    ground_truth = str(task.metadata.get("ground_truth", ""))

    if not ground_truth:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"reason": "no_ground_truth"},
        )

    if _normalized_match(answer_text, ground_truth):
        return EvalOutput(
            reward=1.0,
            is_correct=True,
            signals=[Signal(name="accuracy", value=1.0)],
            metadata={"method": "exact_match"},
        )

    judge_model = task.metadata.get("judge_model") or os.environ.get("RLLM_JUDGE_MODEL")
    judge_base_url = task.metadata.get("judge_base_url") or os.environ.get("RLLM_JUDGE_BASE_URL")
    judge_result = _call_judge(answer_text, ground_truth, judge_model, judge_base_url)
    if judge_result is not None:
        return EvalOutput(
            reward=1.0 if judge_result else 0.0,
            is_correct=judge_result,
            signals=[Signal(name="accuracy", value=1.0 if judge_result else 0.0)],
            metadata={"method": "llm_judge"},
        )

    f1 = _compute_f1(answer_text, ground_truth)
    is_correct = f1 >= 0.8
    return EvalOutput(
        reward=f1,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=f1)],
        metadata={"method": "f1_fallback", "f1": f1},
    )


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _normalized_match(candidate: str, reference: str) -> bool:
    return _normalize(candidate) == _normalize(reference)


def _compute_f1(candidate: str, reference: str) -> float:
    pred = _normalize(candidate).split()
    gold = _normalize(reference).split()
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def _call_judge(candidate: str, reference: str, judge_model: str | None, judge_base_url: str | None) -> bool | None:
    if not judge_base_url:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(base_url=judge_base_url, api_key="EMPTY")
        user_message = _EQUALITY_USER_TEMPLATE.format(reference=reference, candidate=candidate)
        response = client.chat.completions.create(
            model=judge_model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": _EQUALITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        result_text = response.choices[0].message.content or ""
        m = re.search(r"\{[^}]+\}", result_text)
        if m:
            result = json.loads(m.group())
            return bool(result.get("equivalent", False))
        return None
    except Exception as e:
        logger.warning("LLM equality judge call failed: %s", e)
        return None
