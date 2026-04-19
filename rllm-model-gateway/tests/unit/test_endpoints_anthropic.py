"""Anthropic /v1/messages endpoint shaper unit tests."""

from __future__ import annotations

import pytest
from rllm_model_gateway.endpoints import anthropic_messages as am
from rllm_model_gateway.normalized import NormalizedResponse, ToolCall, Usage


def test_system_promoted_from_top_level():
    body = {
        "model": "claude-x",
        "max_tokens": 100,
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "hi"}],
    }
    req = am.to_normalized_request(body)
    assert req.messages[0].role == "system"
    assert req.messages[0].content == "Be brief."
    assert req.messages[1].content == "hi"


def test_system_as_block_list():
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "system": [{"type": "text", "text": "Helper"}],
    }
    req = am.to_normalized_request(body)
    assert req.messages[0].content == "Helper"


def test_content_blocks_collapsed_to_text():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I think..."},
                    {"type": "text", "text": "Hello"},
                ],
            }
        ]
    }
    req = am.to_normalized_request(body)
    assert req.messages[0].reasoning == "I think..."
    assert req.messages[0].content == "Hello"


def test_tool_use_in_assistant_message():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "calc", "input": {"x": 5}},
                ],
            }
        ]
    }
    req = am.to_normalized_request(body)
    tc = req.messages[0].tool_calls[0]
    assert tc.id == "toolu_1"
    assert tc.name == "calc"
    assert tc.arguments == '{"x": 5}'


def test_tool_result_becomes_tool_message():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result text"},
                ],
            }
        ]
    }
    req = am.to_normalized_request(body)
    assert req.messages[0].role == "tool"
    assert req.messages[0].tool_call_id == "toolu_1"
    assert req.messages[0].content == "result text"


def test_tool_schema_renamed():
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "weather", "description": "Get it", "input_schema": {"type": "object", "properties": {}}}],
    }
    req = am.to_normalized_request(body)
    assert req.tools[0].name == "weather"
    assert req.tools[0].parameters == {"type": "object", "properties": {}}


def test_parse_upstream_response_round_trip():
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [
            {"type": "thinking", "thinking": "I think"},
            {"type": "text", "text": "Reply"},
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    resp = am.parse_upstream_response(body)
    assert resp.content == "Reply"
    assert resp.reasoning == "I think"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].arguments == '{"a": 1}'
    assert resp.finish_reason == "tool_calls"


def test_stop_reason_mapping():
    assert am._anthropic_stop_reason_to_normalized("end_turn") == "stop"
    assert am._anthropic_stop_reason_to_normalized("max_tokens") == "length"
    assert am._anthropic_stop_reason_to_normalized("tool_use") == "tool_calls"
    assert am._anthropic_stop_reason_to_normalized("stop_sequence") == "stop"
    assert am._anthropic_stop_reason_to_normalized(None) == "stop"


def test_outbound_nonstream_emits_blocks():
    resp = NormalizedResponse(
        content="hi",
        reasoning="thinking",
        tool_calls=[ToolCall(id="toolu_x", name="t", arguments='{"k":1}')],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=3, completion_tokens=4),
    )
    wire = am.from_normalized_response_nonstream(resp, model="m")
    assert wire["type"] == "message"
    assert wire["role"] == "assistant"
    types = [b["type"] for b in wire["content"]]
    assert types == ["thinking", "text", "tool_use"]
    assert wire["stop_reason"] == "tool_use"
    assert wire["usage"]["input_tokens"] == 3


def test_parse_upstream_stream_event_accumulation():
    chunks = [
        {"type": "message_start", "message": {"id": "msg_x", "usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_y", "name": "f"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a"'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ": 1}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]
    resp = am.parse_upstream_stream(chunks)
    assert resp.content == "Hello world"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].arguments == '{"a": 1}'
    assert resp.finish_reason == "tool_calls"
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 7


@pytest.mark.asyncio
async def test_outbound_stream_event_order():
    resp = NormalizedResponse(
        content="hi",
        reasoning="t",
        tool_calls=[ToolCall(id="x", name="t", arguments="{}")],
        finish_reason="tool_calls",
    )
    events = []
    async for e in am.from_normalized_response_stream(resp, model="m"):
        first = e.split("\n")[0]
        events.append(first[len("event: ") :])
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert "content_block_start" in events
    assert "content_block_delta" in events
    assert "message_delta" in events
