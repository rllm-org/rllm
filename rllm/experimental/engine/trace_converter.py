"""Convert gateway TraceRecord to training-compatible Step."""

from __future__ import annotations

import json
from typing import Any

from rllm_model_gateway.models import TraceRecord

from rllm.agents.agent import Step
from rllm.experimental.rollout import ModelOutput
from rllm.tools.tool_base import ToolCall


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

    model_output = ModelOutput(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        prompt_ids=trace.prompt_token_ids,
        completion_ids=trace.completion_token_ids,
        logprobs=trace.logprobs or [],
        prompt_length=len(trace.prompt_token_ids),
        completion_length=len(trace.completion_token_ids),
        finish_reason=trace.finish_reason,
    )

    # Build chat_completions: input messages + assistant response
    chat_completions = list(trace.messages)
    chat_completions.append(trace.response_message)

    return Step(
        id=trace.trace_id,
        chat_completions=chat_completions,
        model_output=model_output,
        model_response=content,
        thought=reasoning,
        metadata=trace.metadata,
    )
