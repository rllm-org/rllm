"""Legacy completions endpoint shaper unit tests."""

from __future__ import annotations

import json

import pytest
from rllm_model_gateway.endpoints import completions
from rllm_model_gateway.normalized import NormalizedResponse, Usage


def test_request_to_normalized_string_prompt():
    body = {"model": "gpt-3.5-turbo-instruct", "prompt": "Hello", "temperature": 0.5}
    req = completions.to_normalized_request(body)
    assert req.prompt == "Hello"
    assert req.messages is None
    assert req.sampling_params == {"temperature": 0.5}


def test_request_to_normalized_list_prompt_takes_first():
    body = {"prompt": ["first", "second"]}
    req = completions.to_normalized_request(body)
    assert req.prompt == "first"


def test_parse_upstream_response():
    body = {
        "choices": [{"text": "A reply", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    resp = completions.parse_upstream_response(body)
    assert resp.content == "A reply"
    assert resp.finish_reason == "stop"


def test_parse_upstream_stream_concat():
    chunks = [
        {"choices": [{"text": "A "}]},
        {"choices": [{"text": "reply"}]},
        {"choices": [{"text": "", "finish_reason": "stop"}]},
    ]
    resp = completions.parse_upstream_stream(chunks)
    assert resp.content == "A reply"


def test_outbound_nonstream():
    resp = NormalizedResponse(content="hello", finish_reason="length", usage=Usage(prompt_tokens=2, completion_tokens=3))
    wire = completions.from_normalized_response_nonstream(resp, model="m")
    assert wire["object"] == "text_completion"
    assert wire["choices"][0]["text"] == "hello"
    assert wire["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_outbound_stream():
    resp = NormalizedResponse(content="abc", usage=Usage(prompt_tokens=1, completion_tokens=3))
    chunks = []
    async for c in completions.from_normalized_response_stream(resp, model="m"):
        chunks.append(c)
    assert chunks[-1] == "data: [DONE]\n\n"
    body = json.loads(chunks[0][6:].strip())
    assert body["choices"][0]["text"] == "abc"
