"""Convert gateway TraceRecord to training-compatible Step, plus shared metrics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rllm_model_gateway.models import TraceRecord

from rllm.engine.rollout import ModelOutput
from rllm.tools.tool_base import ToolCall
from rllm.types import Step, Trajectory

if TYPE_CHECKING:
    from rllm_model_gateway.v2.types import TraceRecord as V2TraceRecord


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


def trace_record_to_step(trace: TraceRecord | V2TraceRecord) -> Step:
    """Convert a gateway TraceRecord to a training Step.

    TraceRecord has clean top-level fields from vLLM:
    - prompt_token_ids
    - completion_token_ids
    - logprobs (per-token)
    """
    if not isinstance(trace, TraceRecord):
        from rllm_model_gateway.v2.types import TraceRecord as V2TraceRecord

        if not isinstance(trace, V2TraceRecord):
            raise TypeError(f"unsupported gateway trace type: {type(trace).__name__}")
        content = trace.response.content if trace.response.content is not None else trace.response.text
        reasoning = trace.response.reasoning_content or ""
        tool_calls = _parse_openai_tool_calls(trace.response.tool_calls) if trace.response.tool_calls else None
        metadata = dict(trace.output.metadata)
        metadata["request_id"] = trace.request.request_id
        if trace.response.finish_reason is not None:
            metadata["finish_reason"] = trace.response.finish_reason
        model_output = ModelOutput(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_ids=trace.input.prompt_token_ids,
            completion_ids=trace.output.completion_token_ids,
            logprobs=trace.output.logprobs or [],
            routing_matrices=trace.output.routed_experts,
            prompt_length=len(trace.input.prompt_token_ids),
            completion_length=len(trace.output.completion_token_ids),
            finish_reason=trace.response.finish_reason,
            weight_version=trace.output.weight_version,
        )
        response_message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            response_message["reasoning"] = reasoning
        if trace.response.tool_calls:
            response_message["tool_calls"] = trace.response.tool_calls
        chat_completions = list(trace.request.messages)
        chat_completions.append(response_message)
        return Step(
            id=trace.request.request_id,
            chat_completions=chat_completions,
            model_output=model_output,
            model_response=content,
            output=content,
            thought=reasoning,
            metadata=metadata,
            weight_version=trace.output.weight_version,
        )

    content = trace.response_message.get("content", "") or ""
    reasoning = trace.response_message.get("reasoning", "") or ""

    # Extract tool_calls from response message (OpenAI format)
    raw_tool_calls = trace.response_message.get("tool_calls")
    tool_calls = _parse_openai_tool_calls(raw_tool_calls) if raw_tool_calls else None

    model_output = ModelOutput(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        prompt_ids=trace.prompt_token_ids,
        completion_ids=trace.completion_token_ids,
        logprobs=trace.logprobs or [],
        routing_matrices=trace.routing_matrices,
        prompt_length=len(trace.prompt_token_ids),
        completion_length=len(trace.completion_token_ids),
        finish_reason=trace.finish_reason,
        weight_version=trace.weight_version,
    )

    # Build chat_completions: input messages + assistant response
    chat_completions = list(trace.messages)
    chat_completions.append(trace.response_message)

    return Step(
        id=trace.trace_id,
        chat_completions=chat_completions,
        model_output=model_output,
        model_response=content,
        output=content,
        thought=reasoning,
        metadata=trace.metadata,
        weight_version=trace.weight_version,
    )


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
