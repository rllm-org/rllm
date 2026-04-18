"""GatewayClient + AsyncGatewayClient unit tests."""

from __future__ import annotations

import threading
import time

import httpx
import pytest
import uvicorn
from rllm_model_gateway import (
    AsyncGatewayClient,
    GatewayClient,
    GatewayConfig,
    NormalizedRequest,
    NormalizedResponse,
    Usage,
    create_app,
)


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _adapter(_: NormalizedRequest) -> NormalizedResponse:
    return NormalizedResponse(content="ok", finish_reason="stop", usage=Usage(prompt_tokens=1, completion_tokens=1))


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    from rllm_model_gateway.store.memory_store import MemoryTraceStore

    cfg = GatewayConfig(host="127.0.0.1", port=port, admin_api_key="admin", agent_api_key="agent")
    app = create_app(cfg, adapter=_adapter, store=MemoryTraceStore())
    s = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=s.run, daemon=True).start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    s.should_exit = True


def test_health(server):
    c = GatewayClient(server)  # health is public
    assert c.health() == {"status": "ok"}


def test_create_session_and_url(server):
    c = GatewayClient(server, api_key="admin")
    sid = c.create_session("u1", metadata={"k": "v"}, sampling_params={"temperature": 0.5})
    assert sid == "u1"
    assert c.get_session_url("u1") == f"{server}/sessions/u1/v1"
    assert c.get_anthropic_session_url("u1") == f"{server}/sessions/u1"


def test_get_and_list_sessions(server):
    c = GatewayClient(server, api_key="admin")
    c.create_session("u_get", metadata={"x": 1})
    info = c.get_session("u_get")
    assert info["session_id"] == "u_get"
    assert info["metadata"] == {"x": 1}

    sessions = c.list_sessions()
    assert any(s["session_id"] == "u_get" for s in sessions)


def test_get_traces_and_extras(server):
    import uuid

    c = GatewayClient(server, api_key="admin")
    sid = f"u_tr_{uuid.uuid4().hex[:8]}"  # avoid cross-test pollution
    c.create_session(sid)
    httpx.post(f"{server}/sessions/{sid}/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}, headers={"Authorization": "Bearer agent"})
    c.flush()
    traces = c.get_session_traces(sid)
    assert len(traces) == 1
    assert traces[0].endpoint == "chat_completions"

    one = c.get_trace(traces[0].trace_id)
    assert one.trace_id == traces[0].trace_id

    # No extras emitted by this adapter.
    assert c.get_trace_extras(traces[0].trace_id) is None


def test_delete_session(server):
    c = GatewayClient(server, api_key="admin")
    c.create_session("u_del")
    httpx.post(f"{server}/sessions/u_del/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}, headers={"Authorization": "Bearer agent"})
    deleted = c.delete_session("u_del")
    assert deleted == 1
    # Session is gone — get_session() now 404s; raise_for_status throws.
    with pytest.raises(httpx.HTTPStatusError):
        c.get_session("u_del")


def test_flush(server):
    c = GatewayClient(server, api_key="admin")
    assert c.flush() is True


def test_no_api_key_blocks_management(server):
    c = GatewayClient(server)  # no api_key
    with pytest.raises(httpx.HTTPStatusError):
        c.create_session("u_x")


def test_async_client_basic_flow(server):
    import asyncio

    async def run():
        async with AsyncGatewayClient(server, api_key="admin") as c:
            sid = await c.create_session("u_async", sampling_params={"temperature": 0.7})
            info = await c.get_session(sid)
            assert info["session_id"] == sid
            assert (await c.health()) == {"status": "ok"}
            assert await c.flush()

    asyncio.get_event_loop().run_until_complete(run()) if False else asyncio.run(run())
