"""Responses endpoint shaper unit tests.

Covers input flattening (heterogeneous list[InputItem]) and output reshaping
(NormalizedResponse → typed OutputItem list).
"""

from __future__ import annotations

import pytest
from rllm_model_gateway.endpoints import responses
from rllm_model_gateway.normalized import NormalizedResponse, ToolCall, Usage


def test_simple_string_input():
    body = {"model": "m", "input": "Hello there"}
    req = responses.to_normalized_request(body)
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "Hello there"


def test_instructions_become_system():
    body = {"model": "m", "instructions": "You are a helper", "input": "hi"}
    req = responses.to_normalized_request(body)
    assert req.messages[0].role == "system"
    assert req.messages[0].content == "You are a helper"
    assert req.messages[1].role == "user"


def test_input_with_message_items():
    body = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Q1"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "A1"}]},
            {"type": "message", "role": "user", "content": "Q2"},
        ],
    }
    req = responses.to_normalized_request(body)
    assert len(req.messages) == 3
    assert req.messages[0].content == "Q1"
    assert req.messages[1].role == "assistant"
    assert req.messages[1].content == "A1"
    assert req.messages[2].content == "Q2"


def test_input_with_function_call_and_output():
    body = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "What weather?"},
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"SF"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            },
        ],
    }
    req = responses.to_normalized_request(body)
    assert len(req.messages) == 3
    assert req.messages[1].role == "assistant"
    assert req.messages[1].tool_calls[0].name == "get_weather"
    assert req.messages[1].tool_calls[0].arguments == '{"city":"SF"}'
    assert req.messages[2].role == "tool"
    assert req.messages[2].tool_call_id == "call_1"
    assert req.messages[2].content == "sunny"


def test_input_with_reasoning_item():
    body = {
        "model": "m",
        "input": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I considered options"}]},
            {"type": "message", "role": "assistant", "content": "answer"},
        ],
    }
    req = responses.to_normalized_request(body)
    # reasoning becomes its own assistant message with reasoning field
    assert req.messages[0].role == "assistant"
    assert req.messages[0].reasoning == "I considered options"
    assert req.messages[1].content == "answer"


def test_responses_tools_normalized():
    body = {
        "model": "m",
        "input": "hi",
        "tools": [
            {"type": "function", "name": "weather", "description": "Get weather", "parameters": {"type": "object"}},
            {"type": "mcp", "server_label": "x"},  # unsupported, dropped
        ],
    }
    req = responses.to_normalized_request(body)
    assert len(req.tools) == 1
    assert req.tools[0].name == "weather"


def test_previous_response_id_rejected():
    with pytest.raises(ValueError):
        responses.to_normalized_request({"input": "hi", "previous_response_id": "resp_x"})


def test_background_rejected():
    with pytest.raises(ValueError):
        responses.to_normalized_request({"input": "hi", "background": True})


def test_max_output_tokens_translated():
    req = responses.to_normalized_request({"input": "hi", "max_output_tokens": 256})
    assert req.kwargs == {"max_tokens": 256}


def test_reasoning_effort_extracted():
    req = responses.to_normalized_request({"input": "hi", "reasoning": {"effort": "high"}})
    assert req.kwargs["reasoning_effort"] == "high"


def test_outbound_emits_typed_output_items():
    resp = NormalizedResponse(
        content="The answer is 42",
        reasoning="I used calculation",
        tool_calls=[ToolCall(id="call_1", name="calc", arguments='{"x":42}')],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=8),
    )
    wire = responses.from_normalized_response_nonstream(resp, model="m")
    assert wire["status"] == "completed"
    types = [item["type"] for item in wire["output"]]
    assert types == ["reasoning", "message", "function_call"]
    msg_item = wire["output"][1]
    assert msg_item["content"][0]["text"] == "The answer is 42"
    fc_item = wire["output"][2]
    assert fc_item["name"] == "calc"
    assert fc_item["arguments"] == '{"x":42}'


def test_parse_upstream_response_round_trip():
    upstream = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Reply"}]},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a":1}'},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    resp = responses.parse_upstream_response(upstream)
    assert resp.content == "Reply"
    assert resp.reasoning == "thought"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "f"
    assert resp.tool_calls[0].arguments == '{"a":1}'
    assert resp.finish_reason == "tool_calls"
    assert resp.usage.prompt_tokens == 5


def test_parse_upstream_stream_uses_completed_event():
    chunks = [
        {"type": "response.created", "response": {"id": "resp_x"}},
        {"type": "response.output_text.delta", "delta": "Hello "},
        {"type": "response.output_text.delta", "delta": "world"},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello world"}]}],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        },
    ]
    resp = responses.parse_upstream_stream(chunks)
    assert resp.content == "Hello world"
    assert resp.usage.completion_tokens == 3


@pytest.mark.asyncio
async def test_outbound_stream_event_sequence():
    resp = NormalizedResponse(
        content="hi",
        reasoning="thinking",
        tool_calls=[ToolCall(id="c1", name="t", arguments="{}")],
        finish_reason="tool_calls",
    )
    events = []
    async for e in responses.from_normalized_response_stream(resp, model="m"):
        # Each event line: "event: <name>\ndata: <json>\n\n"
        assert e.startswith("event: ")
        first_line = e.split("\n")[0]
        events.append(first_line[len("event: ") :])
    # Check expected ordering pattern
    assert events[0] == "response.created"
    assert events[1] == "response.in_progress"
    assert events[-1] == "response.completed"
    # Reasoning, message, function_call in order
    assert "response.reasoning_summary_text.delta" in events
    assert "response.output_text.delta" in events
    assert "response.function_call_arguments.delta" in events
