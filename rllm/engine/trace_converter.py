"""Convert gateway TraceRecord to training-compatible Step, plus shared metrics."""

import json
from typing import Any

from rllm_model_gateway.models import TraceDelta, TraceRecord

from rllm.engine.rollout import ModelOutput
from rllm.tools.tool_base import ToolCall
from rllm.types import Step, StepDelta, Trajectory


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


def trace_to_model_output(
    trace: TraceRecord | StepDelta,
    response_message: dict[str, Any],
    prompt_ids: list[Any],
    completion_ids: list[int],
    logprobs: list[float] | None,
    routing_matrices: list[str] | None,
) -> ModelOutput:
    """Build the legacy ``ModelOutput`` view from a flat or compact trace."""
    raw_tool_calls = response_message.get("tool_calls")
    return ModelOutput(
        content=response_message.get("content", "") or "",
        reasoning=response_message.get("reasoning", "") or "",
        tool_calls=_parse_openai_tool_calls(raw_tool_calls) if raw_tool_calls else None,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs or [],
        routing_matrices=routing_matrices,
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        finish_reason=trace.finish_reason,
        weight_version=trace.weight_version,
    )


def trace_record_to_step(trace: TraceRecord) -> Step:
    """Convert a gateway TraceRecord to a training Step.

    TraceRecord has clean top-level fields from vLLM:
    - prompt_token_ids
    - completion_token_ids
    - logprobs (per-token)
    """
    model_output = trace_to_model_output(trace, trace.response_message, trace.prompt_token_ids, trace.completion_token_ids, trace.logprobs, trace.routing_matrices)
    content = model_output.content or ""
    reasoning = model_output.reasoning or ""

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

    return Step(
        id=trace.trace_id,
        chat_completions=chat_completions,
        model_output=model_output,
        model_response=content,
        output=content,
        thought=reasoning,
        metadata=metadata,
        weight_version=trace.weight_version,
    )


def trace_delta_to_step_delta(trace: TraceDelta) -> StepDelta:
    """Map the compact gateway contract to the compact training contract."""
    content = trace.response_message.get("content", "") or ""
    reasoning = trace.response_message.get("reasoning", "") or ""
    metadata = trace.metadata | ({"lineage_id": trace.lineage_id} if trace.lineage_id is not None else {})
    return StepDelta(
        id=trace.trace_id,
        parent_step_id=trace.parent_trace_id,
        output=content,
        metadata=metadata,
        prompt_ids_suffix=trace.prompt_ids_suffix,
        chat_completions_suffix=[*trace.messages_suffix, trace.response_message],
        response_ids=trace.completion_token_ids,
        logprobs=trace.logprobs or [],
        routing_matrices=trace.routing_matrices,
        thought=reasoning,
        model_response=content,
        finish_reason=trace.finish_reason,
        weight_version=trace.weight_version,
    )


def compute_step_metrics(trajectories: list[Trajectory]) -> dict:
    """Standard training metrics from trajectories (shared by local and remote engines)."""
    all_response_lens = [len(s.response_ids) for t in trajectories for s in t.steps]
    all_prompt_lens = [len(s.prompt_ids) for t in trajectories for s in t.steps]
    return {
        "traj_per_episode": len(trajectories),
        "steps_used": sum(len(t.steps) for t in trajectories),
        "mean_response_len": (sum(all_response_lens) / len(all_response_lens) if all_response_lens else 0),
        "max_response_len": max(all_response_lens, default=0),
        "min_response_len": min(all_response_lens, default=0),
        "max_prompt_len": max(all_prompt_lens, default=0),
        "min_prompt_len": min(all_prompt_lens, default=0),
    }
