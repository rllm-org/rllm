"""Claw-Eval grader: LLM-as-judge over the agent's full trajectory.

Claw-Eval ships *no* machine-readable rubrics in its public dataset
(`claw-eval/Claw-Eval` rows carry only ``task_id``/``query``/``fixture``/
``language``/``category``; the official 2,159 rubric items and the Pass³
grading harness are not released). So this grader approximates the official
"completion" judgement: an LLM judge reads the task instruction (the row's
``query``) and the agent's full conversation/trajectory, and decides whether
the task was accomplished, returning a 0/1 reward.

This is intentionally a *completion* proxy, not a faithful reproduction of
Claw-Eval's multi-dimensional Pass³ scoring. It exists so that an eval run
fails only on genuine agent mistakes (or missing judge config) — never on
harness/plumbing issues: every error path returns a finite reward with
diagnostic metadata rather than raising.

Judge model resolution (first hit wins):
  1. ``CLAW_EVAL_JUDGE_MODEL`` / ``CLAW_EVAL_JUDGE_BASE_URL`` /
     ``CLAW_EVAL_JUDGE_API_KEY`` env vars (explicit override).
  2. ``task.metadata`` keys ``judge_model`` / ``judge_base_url`` /
     ``judge_api_key`` (stamped by the dataset build script).
  3. The user's rLLM provider config (``~/.rllm/config.json``) routed through
     litellm with the provider's ``litellm_prefix`` — same mechanism the eval
     proxy uses.
"""

from __future__ import annotations

import json
import logging
import os
import re

from rllm.eval.reward_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are an impartial grader for an autonomous-agent benchmark (Claw-Eval).
You are given a TASK an AI agent was asked to perform and the agent's full
TRANSCRIPT (its messages and tool actions). Decide whether the agent actually
accomplished the task.

Grade strictly on task completion:
- Score 1 only if the agent substantively completed what the task asked for
  (produced the requested artifact/answer or performed the requested actions).
- Score 0 if it refused, stalled, hallucinated, produced a wrong/partial
  result, or only described what it would do without doing it.

Respond with ONLY a JSON object: {"score": 0 or 1, "reasoning": "<one sentence>"}."""

JUDGE_USER_TEMPLATE = """\
## Task
{query}

## Agent transcript
{transcript}

## Final answer (if any)
{answer}

Did the agent accomplish the task? Respond with JSON: {{"score": 0 or 1, "reasoning": "..."}}"""

_MAX_TRANSCRIPT_CHARS = 24000


def evaluate(task: Task, episode: Episode) -> EvalOutput:
    query = task.metadata.get("query") or task.metadata.get("rubric") or (task.instruction if isinstance(task.instruction, str) else "") or ""
    transcript = _build_transcript(episode)
    answer = extract_answer_text(episode)

    model, base_url, api_key = _resolve_judge(task)
    if not model:
        # No judge reachable: don't silently pass. Mark as ungraded (reward 0)
        # with a clear signal so the run surfaces the misconfig rather than
        # inflating accuracy.
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="judge_score", value=0.0)],
            metadata={"reason": "no_judge_configured", "ungraded": True},
        )

    score = _call_judge(query, transcript, answer, model, base_url, api_key)
    if score is None:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="judge_score", value=0.0)],
            metadata={"reason": "judge_call_failed", "ungraded": True},
        )

    is_correct = score >= 0.5
    return EvalOutput(
        reward=float(score),
        is_correct=is_correct,
        signals=[Signal(name="judge_score", value=float(score))],
        metadata={"judge_model": model, "category": task.metadata.get("category", "")},
    )


def _build_transcript(episode: Episode) -> str:
    """Render the agent's conversation/trajectory as plain text for the judge."""
    # 1. Cookbook-style explicit conversation artifact.
    conversation = episode.artifacts.get("conversation") if episode.artifacts else None
    msgs: list[str] = []
    if conversation:
        for m in conversation:
            role = str(m.get("role", "")).upper()
            if role == "SYSTEM":
                continue
            msgs.append(f"{role}: {m.get('content', '')}")
        return _truncate("\n".join(msgs))

    # 2. Gateway-reconstructed Steps. Prefer the richest field available.
    if episode.trajectories:
        traj = episode.trajectories[-1]
        for step in traj.steps:
            cc = getattr(step, "chat_completions", None)
            if cc:
                for m in cc:
                    role = str(m.get("role", "")).upper()
                    if role == "SYSTEM":
                        continue
                    content = m.get("content", "")
                    if isinstance(content, list):  # multimodal blocks
                        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                    msgs.append(f"{role}: {content}")
            else:
                inp = getattr(step, "input", None)
                out = getattr(step, "output", None) or getattr(step, "model_response", None)
                if inp:
                    msgs.append(f"USER: {inp}")
                if out:
                    msgs.append(f"ASSISTANT: {out}")
        if msgs:
            return _truncate("\n".join(msgs))

    # 3. Last resort: just the final answer.
    return _truncate(extract_answer_text(episode))


def _truncate(text: str) -> str:
    text = text or ""
    if len(text) <= _MAX_TRANSCRIPT_CHARS:
        return text
    # Keep head and tail — the task setup and the final result both matter.
    head = text[: _MAX_TRANSCRIPT_CHARS // 2]
    tail = text[-_MAX_TRANSCRIPT_CHARS // 2 :]
    return f"{head}\n...[transcript truncated]...\n{tail}"


def _resolve_judge(task: Task) -> tuple[str, str | None, str | None]:
    """Return (model, base_url, api_key). model is litellm-routable.

    base_url/api_key may be None when routing purely by litellm provider prefix.
    """
    # 1. Env override.
    env_model = os.environ.get("CLAW_EVAL_JUDGE_MODEL")
    if env_model:
        return (
            env_model,
            os.environ.get("CLAW_EVAL_JUDGE_BASE_URL"),
            os.environ.get("CLAW_EVAL_JUDGE_API_KEY"),
        )

    # 2. Per-task metadata.
    meta_model = task.metadata.get("judge_model")
    if meta_model:
        return (meta_model, task.metadata.get("judge_base_url"), task.metadata.get("judge_api_key"))

    # 3. rLLM provider config → litellm-prefixed model.
    try:
        from rllm.eval.config import get_provider_info, load_config

        cfg = load_config()
        if cfg.provider == "custom":
            # Already OpenAI-compatible; call it directly.
            return (cfg.model, cfg.base_url or None, cfg.api_key or "EMPTY")
        info = get_provider_info(cfg.provider)
        if info and cfg.model:
            prefix = info.litellm_prefix
            model = f"{prefix}/{cfg.model}" if prefix and not cfg.model.startswith(prefix + "/") else cfg.model
            return (model, None, cfg.api_key or None)
    except Exception:
        logger.debug("claw_eval: could not resolve judge from rLLM config", exc_info=True)

    return ("", None, None)


def _call_judge(
    query: str,
    transcript: str,
    answer: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> float | None:
    user_message = JUDGE_USER_TEMPLATE.format(query=query, transcript=transcript, answer=answer or "(none)")
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    try:
        import litellm

        kwargs: dict = {"model": model, "messages": messages, "temperature": 0.0}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        resp = litellm.completion(**kwargs)
        text = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("claw_eval judge call failed: %s", e)
        return None

    return _parse_score(text)


def _parse_score(text: str) -> float | None:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return float(json.loads(m.group()).get("score", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Bare-number fallback.
    m2 = re.search(r"\b([01](?:\.\d+)?)\b", text)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            pass
    return None
