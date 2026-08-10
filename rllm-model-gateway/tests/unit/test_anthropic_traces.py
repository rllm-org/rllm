"""Anthropic Messages API shapes in trace building.

Anthropic-protocol agents (claude-code) talk to the gateway via LiteLLM's
/v1/messages translation, so responses and SSE streams arrive in Anthropic
format (no ``choices``). The trace builders must still populate
response_message / finish_reason / token_counts, or rLLM discards the traces
as empty and misclassifies the rollout as EmptyCompletion.
"""

from __future__ import annotations

import json

from rllm_model_gateway.data_process import build_trace_record, build_trace_record_from_chunks


def _anthropic_response():
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "z-ai/glm-5.2",
        "content": [
            {"type": "thinking", "thinking": "let me think"},
            {"type": "text", "text": "I will run nmap."},
            {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "nmap -p- target"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1234, "output_tokens": 56},
    }


def test_build_trace_record_anthropic_response():
    trace = build_trace_record("s-1", {"model": "z-ai/glm-5.2", "messages": []}, _anthropic_response(), 12.0)

    assert trace.response_message["role"] == "assistant"
    assert trace.response_message["content"] == "I will run nmap."
    assert trace.response_message["reasoning"] == "let me think"
    tool_calls = trace.response_message["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "bash"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"command": "nmap -p- target"}
    assert trace.finish_reason == "tool_use"
    assert trace.token_counts == {"prompt": 1234, "completion": 56}


def test_build_trace_record_openai_response_unchanged():
    response = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    trace = build_trace_record("s-1", {"model": "m", "messages": []}, response, 1.0)

    assert trace.response_message == {"role": "assistant", "content": "hi"}
    assert trace.finish_reason == "stop"
    assert trace.token_counts == {"prompt": 3, "completion": 2}


def test_build_trace_record_from_anthropic_chunks():
    chunks = [
        {"type": "message_start", "message": {"model": "z-ai/glm-5.2", "usage": {"input_tokens": 100}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ' -la"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 42}},
        {"type": "message_stop"},
    ]
    trace = build_trace_record_from_chunks("s-1", {"model": "z-ai/glm-5.2", "messages": []}, chunks, 25.0)

    assert trace.response_message["content"] == "Hello world"
    assert trace.response_message["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(trace.response_message["tool_calls"][0]["function"]["arguments"]) == {"command": "ls -la"}
    assert trace.finish_reason == "tool_use"
    assert trace.token_counts == {"prompt": 100, "completion": 42}
    assert trace.model == "z-ai/glm-5.2"


def test_build_trace_record_from_anthropic_chunks_malformed_tool_json():
    """glm-5.1 sometimes streams tool args that never become valid JSON; the
    raw text must be preserved instead of crashing the trace builder."""
    chunks = [
        {"type": "message_start", "message": {"model": "z-ai/glm-5.1", "usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"command": "grep -rn \\x invalid'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    trace = build_trace_record_from_chunks("s-1", {"model": "z-ai/glm-5.1", "messages": []}, chunks, 5.0)

    tool_call = trace.response_message["tool_calls"][0]
    assert tool_call["function"]["name"] == "bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"_raw": '{"command": "grep -rn \\x invalid'}
    assert trace.finish_reason == "tool_use"


def test_build_trace_record_from_openai_chunks_unchanged():
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
    ]
    trace = build_trace_record_from_chunks("s-1", {"model": "m", "messages": []}, chunks, 1.0)

    assert trace.response_message["content"] == "hi"
    assert trace.finish_reason == "stop"
    assert trace.token_counts == {"prompt": 5, "completion": 1}
