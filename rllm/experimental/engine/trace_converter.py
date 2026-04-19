"""Convert gateway TraceRecord to a training-compatible Step."""

from __future__ import annotations

import json
from typing import Any

from rllm_model_gateway import TraceRecord

from rllm.agents.agent import Step, Trajectory
from rllm.experimental.rollout import ModelOutput
from rllm.tools.tool_base import ToolCall as RLLMToolCall


def trace_record_to_step(trace: TraceRecord) -> Step:
    """Convert a gateway ``TraceRecord`` to a training ``Step``.

    The trace's ``extras`` field carries token-level data (prompt_ids,
    completion_ids, logprobs, etc.). When extras is None (caller used the
    lightweight ``get_traces`` path) or an empty dict (no extras emitted by
    the adapter), the Step is built without token data.
    """
    extras = trace.extras or {}

    prompt_ids = list(extras.get("prompt_ids") or [])
    completion_ids = list(extras.get("completion_ids") or [])
    logprobs = list(extras.get("logprobs") or [])
    prompt_logprobs = extras.get("prompt_logprobs")
    routing_matrices = extras.get("routing_matrices")

    tool_calls: list[RLLMToolCall] | None = None
    if trace.tool_calls:
        tool_calls = [RLLMToolCall(name=tc.name, arguments=_safe_json_loads(tc.arguments), arguments_raw=tc.arguments) for tc in trace.tool_calls]

    model_output = ModelOutput(
        text=trace.text,
        content=trace.content or "",
        reasoning=trace.reasoning or "",
        tool_calls=tool_calls,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
        prompt_logprobs=list(prompt_logprobs) if prompt_logprobs else None,
        routing_matrices=list(routing_matrices) if routing_matrices else None,
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        finish_reason=trace.finish_reason,
        weight_version=trace.metadata.get("weight_version") if trace.metadata else None,
    )

    # chat_completions = input messages (as dicts) + assistant response (as dict).
    chat_completions: list[dict[str, Any]] = [m.model_dump(exclude_none=True) for m in (trace.messages or [])]
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": trace.content or ""}
    if trace.reasoning:
        assistant_msg["reasoning"] = trace.reasoning
    if trace.tool_calls:
        assistant_msg["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in trace.tool_calls]
    chat_completions.append(assistant_msg)

    return Step(
        id=trace.trace_id,
        chat_completions=chat_completions,
        model_output=model_output,
        model_response=trace.content or "",
        thought=trace.reasoning or "",
        metadata={},
    )


def _safe_json_loads(s: str) -> dict[str, Any]:
    if not s:
        return {}
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compute_step_metrics(trajectories: list[Trajectory]) -> dict:
    """Standard training metrics from trajectories (shared by local and remote engines)."""
    all_response_lens = [len(s.response_ids) for t in trajectories for s in t.steps]
    all_prompt_lens = [len(s.prompt_ids) for t in trajectories for s in t.steps]
    return {
        "num_trajectories": len(trajectories),
        "steps_used": sum(len(t.steps) for t in trajectories),
        "mean_response_len": (sum(all_response_lens) / len(all_response_lens) if all_response_lens else 0),
        "max_response_len": max(all_response_lens, default=0),
        "min_response_len": min(all_response_lens, default=0),
        "max_prompt_len": max(all_prompt_lens, default=0),
        "min_prompt_len": min(all_prompt_lens, default=0),
    }
