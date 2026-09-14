"""GAIA score function: the official quasi-exact-match scorer for GAIA.

The scoring logic (`question_scorer`, `normalize_number_str`, `split_string`,
`normalize_str`) is ported verbatim from the official GAIA scorer
(https://huggingface.co/spaces/gaia-benchmark/leaderboard, Apache-2.0) so that
`rllm eval gaia` reports numbers comparable to the public leaderboard. GAIA
answers are graded by exact match with type-aware normalization: numbers compare
by value (ignoring $/%/, separators), comma/semicolon lists compare element-wise,
and plain strings compare case/space/punctuation-insensitively.
"""

from __future__ import annotations

import re
import string

from rllm.eval.reward_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task

SYSTEM_PROMPT = (
    "You are a general AI assistant. Reason step by step, then finish with your "
    "final answer on its own line in the form: FINAL ANSWER: <answer>. Your answer "
    "should be a number, as few words as possible, or a comma-separated list of "
    "numbers and/or strings. Do not add units unless asked."
)

_ANSWER_PREFIX = "final answer:"


# --- official GAIA scorer (verbatim logic) --------------------------------------------
def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def normalize_number_str(number_str: str) -> float:
    for ch in ["$", "%", ","]:
        number_str = number_str.replace(ch, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(s: str, char_list: list[str] | None = None) -> list[str]:
    pattern = f"[{''.join(char_list or [',', ';'])}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, *, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    """Return True iff `model_answer` matches `ground_truth` under GAIA's rules."""
    # number
    if _is_float(ground_truth):
        return normalize_number_str(model_answer) == float(ground_truth)

    # comma/semicolon-separated list
    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems, strict=False):
            if _is_float(gt_elem):
                comparisons.append(normalize_number_str(ma_elem) == float(gt_elem))
            else:
                comparisons.append(normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False))
        return all(comparisons)

    # plain string
    return normalize_str(model_answer) == normalize_str(ground_truth)


# --- rLLM evaluator -------------------------------------------------------------------
def _strip_answer_prefix(text: str) -> str:
    """Use the text after the last 'FINAL ANSWER:' marker, if the agent emitted one."""
    idx = text.lower().rfind(_ANSWER_PREFIX)
    return text[idx + len(_ANSWER_PREFIX) :].strip() if idx != -1 else text.strip()


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    answer_text = _strip_answer_prefix(str(extract_answer_text(episode)))
    ground_truth = str(task.metadata.get("ground_truth", "") or "")

    if not ground_truth:
        return EvalOutput(reward=0.0, is_correct=False, signals=[Signal(name="accuracy", value=0.0)], metadata={"error": "no ground truth"})

    try:
        correct = question_scorer(answer_text, ground_truth)
    except Exception as exc:  # a malformed answer must never crash the eval
        return EvalOutput(reward=0.0, is_correct=False, signals=[Signal(name="accuracy", value=0.0)], metadata={"error": str(exc)})

    reward = 1.0 if correct else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=correct,
        signals=[Signal(name="accuracy", value=reward)],
        metadata={"model_answer": answer_text, "ground_truth": ground_truth},
    )
