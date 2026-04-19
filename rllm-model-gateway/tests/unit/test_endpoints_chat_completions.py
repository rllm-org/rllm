"""Chat completions endpoint shaper unit tests."""

from __future__ import annotations

import json

import pytest
from rllm_model_gateway.endpoints import chat_completions as cc
from rllm_model_gateway.normalized import NormalizedResponse, ToolCall, Usage


def test_request_to_normalized():
    body = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ],
        "tools": [{"type": "function", "function": {"name": "weather", "description": "Get weather", "parameters": {"type": "object"}}}],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False,
    }
    req = cc.to_normalized_request(body)
    assert req.messages is not None and len(req.messages) == 2
    assert req.messages[0].role == "system"
    assert req.messages[1].content == "hi"
    assert req.tools is not None and req.tools[0].name == "weather"
    assert req.kwargs == {"temperature": 0.7, "max_tokens": 100}


def test_request_tool_call_arguments_round_trip():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"SF"}'},
                    }
                ],
            }
        ]
    }
    req = cc.to_normalized_request(body)
    tc = req.messages[0].tool_calls[0]
    assert tc.name == "weather"
    assert tc.arguments == '{"city":"SF"}'


def test_parse_upstream_response():
    upstream = {
        "id": "chatcmpl-1",
        "model": "vllm-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Here is the answer",
                    "reasoning_content": "I thought about it",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }
    resp = cc.parse_upstream_response(upstream)
    assert resp.content == "Here is the answer"
    assert resp.reasoning == "I thought about it"
    assert resp.finish_reason == "stop"
    assert resp.usage.completion_tokens == 8


def test_parse_upstream_stream_accumulates():
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]
    resp = cc.parse_upstream_stream(chunks)
    assert resp.content == "Hello world"
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 5


def test_parse_upstream_stream_tool_call_accumulation():
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "weather", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"ci'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty":"SF"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    resp = cc.parse_upstream_stream(chunks)
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "c1"
    assert tc.name == "weather"
    assert tc.arguments == '{"city":"SF"}'


def test_from_normalized_response_nonstream_shape():
    resp = NormalizedResponse(
        content="hi",
        reasoning="thinking",
        tool_calls=[ToolCall(id="c1", name="t", arguments='{"x":1}')],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=3, completion_tokens=4),
    )
    wire = cc.from_normalized_response_nonstream(resp, model="m1")
    assert wire["object"] == "chat.completion"
    assert wire["model"] == "m1"
    msg = wire["choices"][0]["message"]
    assert msg["content"] == "hi"
    assert msg["reasoning"] == "thinking"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"x":1}'
    assert wire["choices"][0]["finish_reason"] == "tool_calls"
    assert wire["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_from_normalized_response_stream_emits_done():
    resp = NormalizedResponse(
        content="hi",
        finish_reason="stop",
        usage=Usage(prompt_tokens=3, completion_tokens=2),
    )
    chunks = []
    async for evt in cc.from_normalized_response_stream(resp, model="m1"):
        chunks.append(evt)
    assert chunks[-1] == "data: [DONE]\n\n"
    # Each non-DONE chunk is a `data: <json>\n\n`
    parsed = []
    for c in chunks[:-1]:
        assert c.startswith("data: ")
        parsed.append(json.loads(c[6:].strip()))
    # Roles + content + finish + usage chunks
    assert any(p["choices"] and p["choices"][0]["delta"].get("role") == "assistant" for p in parsed if p["choices"])
    assert any(p["choices"] and p["choices"][0]["delta"].get("content") == "hi" for p in parsed if p["choices"])
    assert any(p.get("usage", {}).get("total_tokens") == 5 for p in parsed)
