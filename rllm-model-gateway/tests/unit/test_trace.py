"""trace.py: build_trace assembly and extras serialization round-trip."""

from __future__ import annotations

from rllm_model_gateway import (
    Message,
    NormalizedRequest,
    NormalizedResponse,
    ToolCall,
    ToolSpec,
    Usage,
    build_trace,
    deserialize_extras,
    serialize_extras,
)


def test_build_trace_assembles_all_fields():
    req = NormalizedRequest(
        messages=[Message(role="user", content="hi")],
        tools=[ToolSpec(name="t", description="d", parameters={"k": 1})],
        kwargs={"temperature": 0.7},
    )
    resp = NormalizedResponse(
        content="answer",
        reasoning="thought",
        tool_calls=[ToolCall(id="c", name="t", arguments='{"x":1}')],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        extras={"completion_ids": [1, 2, 3]},
        metrics={"queue_time_ms": 5.0},
        metadata={"worker_id": "vllm-3"},
    )
    t = build_trace(
        session_id="s1",
        endpoint="chat_completions",
        model="m",
        request=req,
        response=resp,
        gateway_latency_ms=42.0,
    )
    assert t.session_id == "s1"
    assert t.endpoint == "chat_completions"
    assert t.model == "m"
    assert t.messages[0].content == "hi"
    assert t.tools[0].name == "t"
    assert t.kwargs == {"temperature": 0.7}
    assert t.content == "answer"
    assert t.reasoning == "thought"
    assert len(t.tool_calls) == 1
    assert t.finish_reason == "tool_calls"
    assert t.usage.prompt_tokens == 10
    # Adapter metrics + gateway-emitted gateway_latency_ms merged.
    assert t.metrics["queue_time_ms"] == 5.0
    assert t.metrics["gateway_latency_ms"] == 42.0
    assert t.metadata == {"worker_id": "vllm-3"}
    assert t.trace_id
    assert t.timestamp > 0


def test_extras_round_trip_msgpack():
    extras = {
        "prompt_ids": [1, 2, 3],
        "completion_ids": [4, 5, 6, 7],
        "logprobs": [-0.1, -0.2, -0.3, -0.4],
        "prompt_logprobs": [0.0, -0.05, -0.1],
        "routing_matrices": ["abc", "def"],
        "custom_field": {"nested": [1, 2, {"deep": "value"}]},
    }
    blob = serialize_extras(extras)
    assert blob is not None
    fmt, data = blob
    assert fmt == "msgpack"
    decoded = deserialize_extras(fmt, data)
    assert decoded == extras


def test_serialize_extras_returns_none_for_empty():
    assert serialize_extras({}) is None
    assert serialize_extras(None) is None  # type: ignore[arg-type]


def test_deserialize_extras_supports_json_format():
    import json

    data = json.dumps({"a": 1}).encode()
    assert deserialize_extras("json", data) == {"a": 1}
