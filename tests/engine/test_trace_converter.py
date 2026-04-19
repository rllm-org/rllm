"""Tests for trace_converter: trace_record_to_step against the new TraceRecord shape."""

from __future__ import annotations

from rllm_model_gateway import (
    Message,
    NormalizedRequest,
    NormalizedResponse,
    Usage,
    build_trace,
)
from rllm_model_gateway import (
    ToolCall as GatewayToolCall,
)

from rllm.experimental.engine.trace_converter import trace_record_to_step


def _build(*, content="Hi there!", reasoning=None, tool_calls=None, finish_reason="stop", extras=None):
    req = NormalizedRequest(messages=[Message(role="user", content="hello")])
    resp = NormalizedResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=3, completion_tokens=2),
    )
    trace = build_trace(
        session_id="s-001",
        endpoint="chat_completions",
        model="test-model",
        request=req,
        response=resp,
        gateway_latency_ms=12.5,
    )
    if extras is not None:
        trace.extras = extras
    return trace


def _default_extras():
    return {"prompt_ids": [1, 2, 3], "completion_ids": [10, 11], "logprobs": [-0.5, -0.3]}


class TestTraceRecordToStep:
    def test_basic_step(self):
        trace = _build(extras=_default_extras())
        step = trace_record_to_step(trace)

        assert step.id == trace.trace_id
        assert step.model_response == "Hi there!"
        assert step.model_output.content == "Hi there!"
        assert step.model_output.prompt_ids == [1, 2, 3]
        assert step.model_output.completion_ids == [10, 11]
        assert step.model_output.logprobs == [-0.5, -0.3]
        assert step.model_output.tool_calls is None

    def test_step_with_tool_calls(self):
        tcs = [
            GatewayToolCall(id="call_0", name="get_weather", arguments='{"city":"London"}'),
            GatewayToolCall(id="call_1", name="calculate", arguments='{"expr":"2+2"}'),
        ]
        trace = _build(content="", tool_calls=tcs, finish_reason="tool_calls", extras=_default_extras())
        step = trace_record_to_step(trace)

        assert step.model_output.tool_calls is not None
        assert len(step.model_output.tool_calls) == 2
        assert step.model_output.tool_calls[0].name == "get_weather"
        assert step.model_output.tool_calls[0].arguments == {"city": "London"}
        assert step.model_output.tool_calls[1].name == "calculate"
        assert step.model_output.tool_calls[1].arguments == {"expr": "2+2"}
        assert step.model_output.finish_reason == "tool_calls"

    def test_step_with_reasoning(self):
        trace = _build(content="42", reasoning="Let me think...", extras=_default_extras())
        step = trace_record_to_step(trace)
        assert step.thought == "Let me think..."
        assert step.model_output.reasoning == "Let me think..."

    def test_chat_completions_includes_response(self):
        trace = _build(extras=_default_extras())
        step = trace_record_to_step(trace)
        assert len(step.chat_completions) == 2
        assert step.chat_completions[-1]["role"] == "assistant"
        assert step.chat_completions[-1]["content"] == "Hi there!"

    def test_no_tool_calls_means_none(self):
        trace = _build(content="just text", extras=_default_extras())
        step = trace_record_to_step(trace)
        assert step.model_output.tool_calls is None

    def test_no_extras_yields_empty_token_data(self):
        trace = _build(extras=None)  # caller fetched the lightweight trace
        step = trace_record_to_step(trace)
        assert step.model_output.prompt_ids == []
        assert step.model_output.completion_ids == []
        assert step.model_output.logprobs == []

    def test_empty_extras_dict_also_yields_empty_token_data(self):
        trace = _build(extras={})  # asked for extras, adapter emitted none
        step = trace_record_to_step(trace)
        assert step.model_output.prompt_ids == []
        assert step.model_output.completion_ids == []

    def test_optional_extras_fields(self):
        extras = _default_extras()
        extras["prompt_logprobs"] = [-0.1, -0.2, -0.05]
        extras["routing_matrices"] = ["a", "b"]
        trace = _build(extras=extras)
        step = trace_record_to_step(trace)
        assert step.model_output.prompt_logprobs == [-0.1, -0.2, -0.05]
        assert step.model_output.routing_matrices == ["a", "b"]
