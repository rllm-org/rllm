"""LLM-based semantic equality evaluator.

Checks whether a model's answer is semantically equivalent to the ground truth.
Flow: extract answer -> try exact/normalized match -> LLM judge -> F1 fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
from collections import Counter

from rllm.experimental.eval.types import EvalOutput, Signal, _extract_agent_answer
from rllm.types import Episode

logger = logging.getLogger(__name__)

EQUALITY_SYSTEM_PROMPT = """\
You are an impartial judge. Determine whether the candidate answer is semantically \
equivalent to the reference answer. Consider meaning, not exact wording. \
Numbers must match in value. Respond with JSON: {"equivalent": true} or {"equivalent": false}."""

EQUALITY_USER_TEMPLATE = """\
Reference answer: {reference}

Candidate answer: {candidate}

Is the candidate answer semantically equivalent to the reference answer? \
Respond with JSON only."""


class LLMEqualityEvaluator:
    """Evaluates by checking semantic equivalence between model answer and ground truth.

    Evaluation pipeline:
    1. Exact normalized match -> 1.0
    2. LLM judge (if available) -> 0.0 or 1.0
    3. F1 heuristic fallback -> continuous score
    """

    def __init__(
        self,
        judge_model: str | None = None,
        judge_base_url: str | None = None,
    ):
        self.judge_model = judge_model or os.environ.get("RLLM_JUDGE_MODEL")
        self.judge_base_url = judge_base_url or os.environ.get("RLLM_JUDGE_BASE_URL")

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        answer_text = _extract_agent_answer(episode)
        ground_truth = str(task.get("ground_truth", ""))

        if not ground_truth:
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=[Signal(name="accuracy", value=0.0)],
                metadata={"reason": "no_ground_truth"},
            )

        # 1. Try exact normalized match
        if self._normalized_match(answer_text, ground_truth):
            return EvalOutput(
                reward=1.0,
                is_correct=True,
                signals=[Signal(name="accuracy", value=1.0)],
                metadata={"method": "exact_match"},
            )

        # 2. Try LLM judge
        judge_result = self._call_judge(answer_text, ground_truth)
        if judge_result is not None:
            is_correct = judge_result
            return EvalOutput(
                reward=1.0 if is_correct else 0.0,
                is_correct=is_correct,
                signals=[Signal(name="accuracy", value=1.0 if is_correct else 0.0)],
                metadata={"method": "llm_judge"},
            )

        # 3. Fallback to F1 heuristic
        f1 = self._compute_f1(answer_text, ground_truth)
        is_correct = f1 >= 0.8
        return EvalOutput(
            reward=f1,
            is_correct=is_correct,
            signals=[Signal(name="accuracy", value=f1)],
            metadata={"method": "f1_fallback", "f1": f1},
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    def _normalized_match(self, candidate: str, reference: str) -> bool:
        """Check if candidate and reference match after normalization."""
        return self._normalize(candidate) == self._normalize(reference)

    def _compute_f1(self, candidate: str, reference: str) -> float:
        """Compute token-overlap F1 score."""
        pred_tokens = self._normalize(candidate).split()
        gold_tokens = self._normalize(reference).split()
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def _call_judge(self, candidate: str, reference: str) -> bool | None:
        """Call LLM judge to check semantic equivalence. Returns True/False or None."""
        if not self.judge_base_url:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.judge_base_url, api_key="EMPTY")

            user_message = EQUALITY_USER_TEMPLATE.format(reference=reference, candidate=candidate)

            response = client.chat.completions.create(
                model=self.judge_model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EQUALITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
            )

            result_text = response.choices[0].message.content or ""

            # Parse JSON response
            json_match = re.search(r"\{[^}]+\}", result_text)
            if json_match:
                result = json.loads(json_match.group())
                return bool(result.get("equivalent", False))

            return None

        except Exception as e:
            logger.warning("LLM equality judge call failed: %s", e)
            return None
