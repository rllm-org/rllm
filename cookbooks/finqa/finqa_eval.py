"""FinQA evaluator: judge LLM correctness + table-access bonus.

Ports the judge call, single-table vs multi-table rubric, and table-access
scoring from the legacy :func:`projects.finqa.fin_qa_reward.fin_qa_reward_function`
into a self-contained module so the cookbook does not depend on
``projects/finqa``.

The judge calls require ``OPENAI_API_KEY``. If it's missing the evaluator
falls back to ``reward=0`` rather than raising — useful for smoke tests
without a real key.
"""

from __future__ import annotations

import json
import os
import re
import time

import openai
from finqa_constants import CORRECTNESS_PROMPT_PATH, MULTI_TABLE_CORRECTNESS_PROMPT_PATH

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task

# ---------------------------------------------------------------------------
# Judge config
# ---------------------------------------------------------------------------

with open(CORRECTNESS_PROMPT_PATH, encoding="utf-8") as f:
    CORRECTNESS_PROMPT = f.read()

with open(MULTI_TABLE_CORRECTNESS_PROMPT_PATH, encoding="utf-8") as f:
    MULTI_TABLE_CORRECTNESS_PROMPT = f.read()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
JUDGE_MODEL = os.environ.get("FINQA_JUDGE_MODEL", "gpt-5.4-nano")
MULTI_TABLE_JUDGE_MODEL = os.environ.get("FINQA_MULTI_TABLE_JUDGE_MODEL", "gpt-5.4-mini")

CORRECTNESS_WEIGHTS = {
    "primary_data_score": 0.30,
    "derived_metrics_score": 0.30,
    "reasoning_score": 0.15,
    "consistency_score": 0.10,
    "completeness_score": 0.10,
    "structure_score": 0.05,
}

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)


def _make_judge_client():
    """Build a plain OpenAI client. Returns None if OPENAI_API_KEY is missing."""
    if not OPENAI_API_KEY:
        return None
    try:
        return openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


_JUDGE_CLIENT = _make_judge_client()

# ---------------------------------------------------------------------------
# Judge call reliability
# ---------------------------------------------------------------------------
# A response that stops for any reason other than a clean completion is not a
# verdict. Measured on Qwen3.8-27B: 12% of calls came back with
# incomplete_details.reason == "max_output_tokens", and in that state the
# Responses API returns an EMPTY output_text -- which this module's
# `("true" in text) and ("false" not in text)` check turns deterministically
# into False, i.e. reward 0. That is a one-directional bias: it can only mark a
# correct answer wrong, never the reverse. So retry instead of scoring it.
JUDGE_MAX_ATTEMPTS = int(os.environ.get("FINQA_JUDGE_MAX_ATTEMPTS", "10"))
JUDGE_FINISH_LOG = os.environ.get("FINQA_JUDGE_FINISH_LOG", "")


def _finish_reason(response) -> str:
    """chat-completions `finish_reason` equivalent for the Responses API."""
    inc = getattr(response, "incomplete_details", None)
    reason = getattr(inc, "reason", None) if inc else None
    status = getattr(response, "status", None)
    if reason:
        return reason          # max_output_tokens | content_filter | ...
    if status == "completed":
        return "stop"
    return status or "unknown"


def _log_finish(finish: str, attempt: int, ok: bool) -> None:
    """Record the finish reason of every judge call."""
    if JUDGE_FINISH_LOG:
        try:
            with open(JUDGE_FINISH_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"{time.time():.3f}\t{finish}\tattempt={attempt}\tok={int(ok)}\n")
        except OSError:
            pass
    if not ok:
        print(f"[finqa-judge] finish_reason={finish} on attempt {attempt}/{JUDGE_MAX_ATTEMPTS}; retrying", flush=True)



def _extract_final_answer(action: str, *, prefer_tail: bool = False) -> str:
    """Pull the FINAL ANSWER section out of a model response."""
    if not action:
        return ""
    m = _FINAL_ANSWER_CODE_BLOCK_RE.search(action)
    if m:
        return m.group(1).strip()
    if not prefer_tail:
        m = _FINAL_ANSWER_PARAGRAPH_RE.search(action)
        if m:
            return m.group(1).strip()
    m = _FINAL_ANSWER_TAIL_RE.search(action)
    if m:
        return m.group(1).strip()
    return action


def _table_access_score(accessed: list[str], expected: str | list[str]) -> float:
    """Fraction of required tables the agent inspected via get_table_info."""
    if not accessed or not expected:
        return 0.0
    seen = {t.lower().strip() for t in accessed if isinstance(t, str) and t.strip()}
    if isinstance(expected, list):
        wanted = [n.lower().strip() for n in expected if isinstance(n, str) and n.strip()]
    else:
        wanted = [expected.lower().strip()] if isinstance(expected, str) else []
    if not wanted:
        return 0.0
    return sum(1 for n in wanted if n in seen) / len(wanted)


def _call_judge(system_prompt: str, user_prompt: str, *, multi_table: bool) -> tuple[bool | float, dict]:
    """Score ``user_prompt`` against ``system_prompt`` via the judge LLM.

    Returns ``(decision, raw_rubric)``. ``decision`` is ``bool`` for
    single-table (true/false judge) and ``float in [0, 1]`` for the
    multi-table weighted rubric. Falls back to ``False``/``0.0`` if the
    judge client is unavailable.
    """
    if _JUDGE_CLIENT is None:
        return (0.0 if multi_table else False), {}

    request: dict = {
        "model": MULTI_TABLE_JUDGE_MODEL if multi_table else JUDGE_MODEL,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": 5000 if multi_table else 512,
    }

    if multi_table:
        schema = {
            "type": "object",
            "properties": {k: {"type": "number"} for k in CORRECTNESS_WEIGHTS} | {"explanation": {"type": "string"}},
            "required": list(CORRECTNESS_WEIGHTS) + ["explanation"],
            "additionalProperties": False,
        }
        request["reasoning"] = {"effort": "medium"}
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "finqa_multi_table_rubric",
                "schema": schema,
                "strict": True,
            }
        }
    else:
        request["reasoning"] = {"effort": "low"}
        request["text"] = {"verbosity": "low"}

    out = ""
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        try:
            response = _JUDGE_CLIENT.responses.create(**request)
            finish = _finish_reason(response)
            out = getattr(response, "output_text", "") or ""
            ok = finish == "stop" and bool(out.strip())
        except Exception as e:  # noqa: BLE001 - any transport/API error is retryable
            finish, out, ok = f"exception:{type(e).__name__}", "", False
        _log_finish(finish, attempt, ok)
        if ok:
            break
    else:
        print(
            f"[finqa-judge] ALARM: {JUDGE_MAX_ATTEMPTS} consecutive abnormal finish_reasons "
            f"(last={finish}); scoring this sample as incorrect. model={request['model']} "
            f"max_output_tokens={request['max_output_tokens']}",
            flush=True,
        )
        return (0.0 if multi_table else False), {}

    try:
        if multi_table:
            try:
                parsed = json.loads(out)
            except json.JSONDecodeError:
                parsed = {}
            weighted, total = 0.0, 0.0
            for k, w in CORRECTNESS_WEIGHTS.items():
                v = parsed.get(k)
                if isinstance(v, int | float):
                    weighted += (float(v) / 100.0) * w
                    total += w
            score = max(0.0, min(1.0, weighted / total if total > 0 else 0.0))
            return score, parsed

        text = out.lower()
        return (("true" in text) and ("false" not in text)), {}
    except Exception:
        return (0.0 if multi_table else False), {}


def _task_meta(task: Task | dict) -> dict:
    """Normalize Runner-supplied Task and trainer-supplied dict to a single shape."""
    if isinstance(task, Task):
        return task.metadata or {}
    return task or {}


@rllm.evaluator
def finqa_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    """Judge-LLM correctness for the agent's FINAL ANSWER, plus table-access bonus."""
    meta = _task_meta(task)
    answer = str(episode.artifacts.get("answer", ""))
    accessed = episode.artifacts.get("accessed_tables", []) or []

    question = meta.get("question")
    core_question = meta.get("core_question") or question
    ground_truth = meta.get("ground_truth")
    qtype = (meta.get("question_type") or "").lower()

    if not answer or not question or not ground_truth:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"reason": "missing answer/question/ground_truth"},
        )

    multi_table = qtype.startswith("multi_table")
    if multi_table:
        prompt_input = f"question : {core_question}\nmodel response : {answer}\nlabel : {ground_truth}"
        verdict, rubric = _call_judge(MULTI_TABLE_CORRECTNESS_PROMPT, prompt_input, multi_table=True)
        correctness = float(verdict)
        is_correct = correctness >= 0.9
    else:
        final_answer = _extract_final_answer(answer)
        prompt_input = f"question : {question}\nmodel response : {final_answer}\nlabel : {ground_truth}"
        verdict, rubric = _call_judge(CORRECTNESS_PROMPT, prompt_input, multi_table=False)
        is_correct = bool(verdict)
        correctness = 1.0 if is_correct else 0.0

    table_bonus = _table_access_score(accessed, meta.get("table_name", ""))

    signals = [
        Signal(name="accuracy", value=correctness),
        Signal(name="table_access", value=table_bonus),
    ]
    metadata: dict = {"right_table_access_reward": table_bonus}
    if multi_table:
        for k in CORRECTNESS_WEIGHTS:
            v = rubric.get(k)
            if isinstance(v, int | float):
                metadata[f"multi_table_{k}"] = float(v)
        metadata["multi_table_overall_score"] = correctness

    return EvalOutput(
        reward=correctness,
        is_correct=is_correct,
        signals=signals,
        metadata=metadata,
    )


# Re-export for tests
__all__ = ["finqa_evaluator", "_extract_final_answer", "_table_access_score"]
