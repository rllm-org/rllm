"""Evaluator for multimodal_codex tasks.

Codex CLI is invoked with ``--json --enable unified_exec``, which emits one
JSON event per line. We locate the last ``agent_message`` event and treat its
content as the model's answer. If JSON parsing fails (schema drift, non-JSON
stdout), we fall back to substring-matching the raw text — same idea as
``sandbox_code_evaluator``.

Answer normalization is aggressive: strip whitespace, remove trailing period,
casefold. Ground truth from ``tasks.py`` is a plain digit string or short
word, so exact match after normalization is the right comparison.
"""

from __future__ import annotations

import json
import re

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task


def _extract_final_content(step_output: str) -> str | None:
    """Parse Codex --json stream: find last ``agent_message`` message content.

    Returns None if no agent_message is found (schema drift OR unparseable stdout).
    Caller should fall back to substring search on raw text.
    """
    if not step_output:
        return None
    last_content: str | None = None
    for line in step_output.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Codex event schema:
        #   {"type": "agent_message", "message": {"role": "assistant", "content": "..."}}
        # OR the flatter schema seen in some versions:
        #   {"type": "agent_message", "content": "..."}
        if ev.get("type") != "agent_message":
            continue
        content = ev.get("content")
        if content is None and isinstance(ev.get("message"), dict):
            content = ev["message"].get("content")
        if isinstance(content, str) and content.strip():
            last_content = content
    return last_content


def _normalize(text: str) -> str:
    """Strip, remove trailing punctuation, casefold — for exact-answer compare."""
    text = text.strip()
    text = re.sub(r"[.,!?;:]+$", "", text)
    return text.casefold()


def _exact_match(model_answer: str, ground_truth: str) -> bool:
    """Primary path: normalized string equality only. Rejects negations
    ("not 3, but 5"), qualifiers, and any answer that isn't literally the GT."""
    return _normalize(model_answer) == _normalize(ground_truth)


def _substring_hit(text: str, ground_truth: str) -> bool:
    """Fallback path ONLY: word-boundary substring. Used when Codex --json
    parsing found no ``agent_message`` — we still want to reward runs where
    the answer appears in the raw stdout (e.g. Codex emits a bare line before
    the wrapper closes)."""
    gt = re.escape(_normalize(ground_truth))
    return bool(re.search(rf"\b{gt}\b", _normalize(text)))


# Legacy public name kept for tests that import ``_matches`` directly. Uses
# the substring semantics (matches either exact or word-boundary).
def _matches(model_answer: str, ground_truth: str) -> bool:
    return _exact_match(model_answer, ground_truth) or _substring_hit(model_answer, ground_truth)


@rllm.evaluator
def multimodal_codex_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    meta = task.metadata if isinstance(task, Task) else task or {}
    ground_truth = str(meta.get("ground_truth", "")).strip()
    task_type = meta.get("task_type", "unknown")

    if not ground_truth:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"error": "no ground_truth in task.metadata", "task_type": task_type},
        )

    # Primary: sweep every step across every trajectory; keep the LAST
    # parseable ``agent_message`` content (Codex may emit intermediate
    # "let me think..." messages before the final answer in later steps).
    model_answer: str | None = None
    for traj in episode.trajectories:
        for step in traj.steps:
            content = _extract_final_content(str(step.output or ""))
            if content is not None:
                model_answer = content  # cross-step last-wins

    if model_answer is not None:
        # Primary path: strict equality only. No substring — negations like
        # "not 3, but 5" must NOT count as gt="3" correct.
        is_correct = _exact_match(model_answer, ground_truth)
    else:
        # Fallback: no agent_message anywhere. Try substring match on raw
        # stdout at word boundary (guards against "5" ⊂ "15").
        for traj in episode.trajectories:
            for step in traj.steps:
                text = str(step.output or "")
                if _substring_hit(text, ground_truth):
                    model_answer = text
                    break
            if model_answer is not None:
                break
        if model_answer is None:
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=[Signal(name="accuracy", value=0.0)],
                metadata={"error": "no answer extracted from episode", "task_type": task_type, "ground_truth": ground_truth},
            )
        is_correct = True  # by construction of _substring_hit

    return EvalOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=1.0 if is_correct else 0.0)],
        metadata={
            "task_type": task_type,
            "ground_truth": ground_truth,
            "model_answer": model_answer[:200],
        },
    )
