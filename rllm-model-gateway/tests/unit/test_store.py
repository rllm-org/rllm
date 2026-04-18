"""TraceStore tests — both SQLite and memory implementations."""

from __future__ import annotations

import os
import tempfile

import pytest
from rllm_model_gateway import (
    Message,
    NormalizedRequest,
    NormalizedResponse,
    ToolCall,
    ToolSpec,
    TraceRecord,
    Usage,
    build_trace,
    serialize_extras,
)
from rllm_model_gateway.store.memory_store import MemoryTraceStore
from rllm_model_gateway.store.sqlite_store import SqliteTraceStore


@pytest.fixture(params=["memory", "sqlite"])
async def store(request):
    if request.param == "memory":
        s = MemoryTraceStore()
        yield s
        await s.close()
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        s = SqliteTraceStore(db_path=tmp.name)
        yield s
        await s.close()
        os.unlink(tmp.name)


def _make_trace(session_id: str = "s1", endpoint: str = "chat_completions") -> TraceRecord:
    req = NormalizedRequest(
        messages=[Message(role="user", content="hi")],
        tools=[ToolSpec(name="t", description="d", parameters={})],
        sampling_params={"temperature": 0.7},
    )
    resp = NormalizedResponse(
        content="hello",
        reasoning="thinking",
        tool_calls=[ToolCall(id="c1", name="t", arguments={"x": 1})],
        finish_reason="stop",
        usage=Usage(prompt_tokens=2, completion_tokens=3),
    )
    return build_trace(
        session_id=session_id,
        endpoint=endpoint,
        model="m",
        request=req,
        response=resp,
        gateway_latency_ms=12.5,
    )


@pytest.mark.asyncio
async def test_create_and_get_session(store):
    await store.create_session("s1", metadata={"k": "v"}, sampling_params={"temperature": 0.7})
    info = await store.get_session("s1")
    assert info["session_id"] == "s1"
    assert info["metadata"] == {"k": "v"}
    assert info["sampling_params"] == {"temperature": 0.7}
    assert info["trace_count"] == 0


@pytest.mark.asyncio
async def test_store_and_get_trace_round_trip(store):
    await store.create_session("s1")
    trace = _make_trace()
    await store.store_trace(trace)
    fetched = await store.get_trace(trace.trace_id)
    assert fetched is not None
    assert fetched.content == "hello"
    assert fetched.reasoning == "thinking"
    assert fetched.messages[0].content == "hi"
    assert len(fetched.tool_calls) == 1
    assert fetched.tool_calls[0].arguments == {"x": 1}


@pytest.mark.asyncio
async def test_extras_split_storage(store):
    await store.create_session("s1")
    trace = _make_trace()
    extras = serialize_extras({"completion_ids": [1, 2, 3], "logprobs": [-0.1, -0.2, -0.3]})
    await store.store_trace(trace, extras=extras)

    traces = await store.get_traces("s1")
    assert len(traces) == 1
    blob = await store.get_trace_extras(trace.trace_id)
    assert blob is not None
    fmt, data = blob
    assert fmt == "msgpack"
    assert isinstance(data, bytes) and len(data) > 0


@pytest.mark.asyncio
async def test_get_traces_orders_by_timestamp(store):
    await store.create_session("s1")
    t1 = _make_trace()
    t2 = _make_trace()
    t1.timestamp = 100.0
    t2.timestamp = 200.0
    await store.store_trace(t2)
    await store.store_trace(t1)
    traces = await store.get_traces("s1")
    assert traces[0].timestamp == 100.0
    assert traces[1].timestamp == 200.0


@pytest.mark.asyncio
async def test_delete_session_cascades(store):
    await store.create_session("s1")
    trace = _make_trace()
    await store.store_trace(trace, extras=serialize_extras({"x": 1}))
    deleted = await store.delete_session("s1")
    assert deleted == 1
    assert await store.get_session("s1") is None
    assert await store.get_trace(trace.trace_id) is None
    assert await store.get_trace_extras(trace.trace_id) is None
