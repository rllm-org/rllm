"""Convert gateway TraceRecord to training-compatible Step, plus shared metrics."""

import json
from typing import Any

from rllm_model_gateway.models import TraceRecord

from rllm.engine.rollout import ModelOutput
from rllm.tools.tool_base import ToolCall
from rllm.types import Step, Trajectory


def is_empty_response_trace(trace: TraceRecord) -> bool:
    """Return whether a stored request attempt produced no model response.

    Transient HTTP failures can be persisted as traces even though they have
    no assistant response envelope.  They are request-attempt diagnostics, not
    model turns, and must be discarded before strict token validation.  A
    response with empty text is still valid when it has an assistant envelope
    or a finish reason (for example, a tool-only turn).
    """
    return not trace.response_message and trace.finish_reason is None and not trace.completion_token_ids


def filter_empty_response_traces(traces: list[TraceRecord]) -> list[TraceRecord]:
    """Keep only traces that contain evidence of a model response."""
    return [trace for trace in traces if not is_empty_response_trace(trace)]


def _parse_openai_tool_calls(raw_tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert OpenAI-format tool_calls to rLLM ToolCall objects."""
    result = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_raw = func.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw)
            except (json.JSONDecodeError, ValueError):
                arguments = {"raw": args_raw}
        else:
            arguments = args_raw
        result.append(ToolCall(name=name, arguments=arguments))
    return result


def trace_record_to_step(trace: TraceRecord) -> Step:
    """Convert a gateway TraceRecord to a training Step.

    TraceRecord has clean top-level fields from vLLM:
    - prompt_token_ids
    - completion_token_ids
    - logprobs (per-token)
    """
    content = trace.response_message.get("content", "") or ""
    reasoning = trace.response_message.get("reasoning", "") or ""

    # Extract tool_calls from response message (OpenAI format)
    raw_tool_calls = trace.response_message.get("tool_calls")
    tool_calls = _parse_openai_tool_calls(raw_tool_calls) if raw_tool_calls else None

    # In compact mode the client keeps prompt ids as the step-form delta
    # marker {"__prompt_ids_delta__": [lcp, suffix]} instead of the full
    # list — the marker rides on step.prompt_ids and model_output carries
    # only the exact length; nothing here materializes the expansion.
    raw_prompt_ids = trace.prompt_token_ids
    ids_delta = raw_prompt_ids.get("__prompt_ids_delta__") if isinstance(raw_prompt_ids, dict) else None
    if ids_delta is not None:
        lcp, suffix = ids_delta
        full_prompt_ids, prompt_length = None, lcp + len(suffix)
    else:
        full_prompt_ids, prompt_length = raw_prompt_ids, len(raw_prompt_ids)

    model_output = ModelOutput(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        prompt_ids=full_prompt_ids,
        completion_ids=trace.completion_token_ids,
        logprobs=trace.logprobs or [],
        routing_matrices=trace.routing_matrices,
        prompt_length=prompt_length,
        completion_length=len(trace.completion_token_ids),
        finish_reason=trace.finish_reason,
        weight_version=trace.weight_version,
    )

    # Build chat_completions: input messages + assistant response
    chat_completions = list(trace.messages)
    chat_completions.append(trace.response_message)

    # Carry the gateway-assigned lineage id (parent vs subagent conversation)
    # onto the step so episode enrichment can split a session's steps into one
    # trajectory per lineage. None when cumulative mode is off (no slots).
    metadata = dict(trace.metadata or {})
    lineage_id = getattr(trace, "lineage_id", None)
    if lineage_id is not None:
        metadata["lineage_id"] = lineage_id

    # CONTRACT (review): the assigned message dicts are SHARED across every
    # Step whose conversation contains them. They must be treated as
    # immutable — mutating one through any Step mutates the shared prefix of
    # all later Steps. No engine/trainer path mutates them today; a frozen
    # history type is the planned enforcement (native-compact follow-up).
    step = Step(
        id=trace.trace_id,
        prompt_ids=raw_prompt_ids if ids_delta is not None else [],
        # Assigned below, AFTER construction: pydantic field validation copies
        # each message dict, which would re-materialize the full conversation
        # prefix per step even with the deepcopy skipped (audit: the root
        # amplifier). Plain attribute assignment keeps the shared references.
        chat_completions=[],
        model_output=model_output,
        model_response=content,
        output=content,
        thought=reasoning,
        metadata=metadata,
        weight_version=trace.weight_version,
    )
    step.chat_completions = chat_completions
    return step


def compute_step_metrics(trajectories: list[Trajectory]) -> dict:
    """Standard training metrics from trajectories (shared by local and remote engines)."""
    all_response_lens = [len(s.response_ids) for t in trajectories for s in t.steps]
    all_prompt_lens = [s.prompt_len for t in trajectories for s in t.steps]
    return {
        "traj_per_episode": len(trajectories),
        "steps_used": sum(len(t.steps) for t in trajectories),
        "mean_response_len": (sum(all_response_lens) / len(all_response_lens) if all_response_lens else 0),
        "max_response_len": max(all_response_lens, default=0),
        "min_response_len": min(all_response_lens, default=0),
        "max_prompt_len": max(all_prompt_lens, default=0),
        "min_prompt_len": min(all_prompt_lens, default=0),
    }
