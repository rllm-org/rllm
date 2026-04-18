"""Passthrough mode: bytes-in, bytes-out forwarding to an upstream URL.

Each test exercises one endpoint. Upstream URL convention follows the SDK
of the upstream being forwarded to:
  - OpenAI-flavored (chat/completions/responses): upstream_url ends in /v1
  - Anthropic-flavored:                            upstream_url has no /v1
"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from rllm_model_gateway import GatewayConfig, create_app
from rllm_model_gateway.store.memory_store import MemoryTraceStore


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def upstream():
    """Fake upstream serving both OpenAI- and Anthropic-flavored routes."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("stream"):

            async def gen():
                yield 'data: {"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":"hi from upstream"},"finish_reason":null}]}\n\n'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "id": "chatcmpl-up",
            "object": "chat.completion",
            "model": body.get("model", "fake"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi from upstream"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    @app.post("/v1/completions")
    async def cmpl(request: Request):
        body = await request.json()
        return {
            "id": "cmpl-up",
            "object": "text_completion",
            "model": body.get("model"),
            "choices": [{"text": "hi from upstream", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    @app.post("/v1/responses")
    async def resp(request: Request):
        body = await request.json()
        return {
            "id": "resp-up",
            "object": "response",
            "status": "completed",
            "model": body.get("model"),
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi from upstream"}]}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    @app.post("/v1/messages")
    async def msgs(request: Request):
        body = await request.json()
        return {
            "id": "msg-up",
            "type": "message",
            "role": "assistant",
            "model": body.get("model"),
            "content": [{"type": "text", "text": "hi from upstream"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/openapi.json", timeout=0.5).status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    yield port
    server.should_exit = True


def _gw_openai(port: int, **overrides) -> TestClient:
    """Gateway pointing at OpenAI-flavored upstream (URL ends in /v1)."""
    cfg = GatewayConfig(
        upstream_url=f"http://127.0.0.1:{port}/v1",
        admin_api_key="k",
        agent_api_key="k",
        **overrides,
    )
    return TestClient(create_app(cfg, store=MemoryTraceStore()), headers={"Authorization": "Bearer k"})


def _gw_anthropic(port: int, **overrides) -> TestClient:
    """Gateway pointing at Anthropic-flavored upstream (URL has no /v1)."""
    cfg = GatewayConfig(
        upstream_url=f"http://127.0.0.1:{port}",
        admin_api_key="k",
        agent_api_key="k",
        **overrides,
    )
    return TestClient(create_app(cfg, store=MemoryTraceStore()), headers={"Authorization": "Bearer k"})


def _flush_and_get_traces(gw: TestClient, sid: str) -> list:
    gw.post("/admin/flush")
    return gw.get(f"/sessions/{sid}/traces").json()


def test_passthrough_chat_completions(upstream):
    with _gw_openai(upstream) as gw:
        gw.post("/sessions", json={"session_id": "s1"})
        r = gw.post("/sessions/s1/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["message"]["content"] == "hi from upstream"
        traces = _flush_and_get_traces(gw, "s1")
        assert traces[0]["content"] == "hi from upstream"
        assert traces[0]["endpoint"] == "chat_completions"


def test_passthrough_completions(upstream):
    with _gw_openai(upstream) as gw:
        gw.post("/sessions", json={"session_id": "s2"})
        r = gw.post("/sessions/s2/v1/completions", json={"model": "m", "prompt": "hi"})
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["text"] == "hi from upstream"
        traces = _flush_and_get_traces(gw, "s2")
        assert traces[0]["endpoint"] == "completions"
        assert traces[0]["content"] == "hi from upstream"


def test_passthrough_responses(upstream):
    with _gw_openai(upstream) as gw:
        gw.post("/sessions", json={"session_id": "s3"})
        r = gw.post("/sessions/s3/v1/responses", json={"model": "m", "input": "hi"})
        assert r.status_code == 200, r.text
        assert r.json()["output"][0]["content"][0]["text"] == "hi from upstream"
        traces = _flush_and_get_traces(gw, "s3")
        assert traces[0]["endpoint"] == "responses"


def test_passthrough_anthropic(upstream):
    with _gw_anthropic(upstream) as gw:
        gw.post("/sessions", json={"session_id": "s4"})
        r = gw.post("/sessions/s4/v1/messages", json={"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        assert r.json()["content"][0]["text"] == "hi from upstream"
        traces = _flush_and_get_traces(gw, "s4")
        assert traces[0]["endpoint"] == "anthropic_messages"


def test_passthrough_chat_completions_streaming(upstream):
    with _gw_openai(upstream) as gw:
        gw.post("/sessions", json={"session_id": "s5"})
        with gw.stream("POST", "/sessions/s5/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
            assert r.status_code == 200
            data_lines = [line for line in r.iter_lines() if line.startswith("data: ")]
            assert "data: [DONE]" in data_lines

        traces = _flush_and_get_traces(gw, "s5")
        assert len(traces) == 1
        assert traces[0]["content"] == "hi from upstream"


def test_passthrough_overrides_model_and_sampling():
    """Verify model pin + session sampling apply in passthrough mode too."""
    received: list = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        received.append(await request.json())
        return {
            "id": "x",
            "object": "chat.completion",
            "model": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/openapi.json", timeout=0.5).status_code == 200:
                break
        except Exception:
            time.sleep(0.05)

    try:
        with _gw_openai(port, model="pinned-model", sampling_params_priority="session") as gw:
            gw.post(
                "/sessions",
                json={
                    "session_id": "s1",
                    "sampling_params": {"temperature": 0.42, "max_tokens": 99},
                },
            )
            gw.post(
                "/sessions/s1/v1/chat/completions",
                json={
                    "model": "client-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.99,
                },
            )
        assert received, "upstream never received a request"
        sent = received[-1]
        assert sent["model"] == "pinned-model"
        assert sent["temperature"] == 0.42
        assert sent["max_tokens"] == 99
    finally:
        server.should_exit = True


def test_passthrough_strips_agent_auth_header_and_injects_upstream_key():
    """Agent's Authorization header must NOT leak to the upstream;
    upstream_api_key (when set) is sent instead."""
    received: list = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        received.append({k: v for k, v in request.headers.items()})
        return {
            "id": "x",
            "object": "chat.completion",
            "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/openapi.json", timeout=0.5).status_code == 200:
                break
        except Exception:
            time.sleep(0.05)

    try:
        cfg = GatewayConfig(
            upstream_url=f"http://127.0.0.1:{port}/v1",
            upstream_api_key="UPSTREAM-SECRET",
            admin_api_key="k",
            agent_api_key="agent-k",
        )
        with TestClient(create_app(cfg, store=MemoryTraceStore()), headers={"Authorization": "Bearer k"}) as gw:
            gw.post("/sessions", json={"session_id": "s1"})
            # Pretend agent posts with the agent_api_key.
            gw.post("/sessions/s1/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}, headers={"Authorization": "Bearer agent-k"})

        assert received
        upstream_headers = received[-1]
        # Agent's bearer is NOT what the upstream sees.
        assert upstream_headers.get("authorization") == "Bearer UPSTREAM-SECRET"
        # x-api-key also injected for Anthropic-style upstreams.
        assert upstream_headers.get("x-api-key") == "UPSTREAM-SECRET"
    finally:
        server.should_exit = True


def test_passthrough_retries_on_connect_error():
    cfg = GatewayConfig(
        upstream_url="http://127.0.0.1:1/v1",  # nothing listening
        admin_api_key="k",
        agent_api_key="k",
        max_retries=1,
    )
    with TestClient(create_app(cfg, store=MemoryTraceStore()), headers={"Authorization": "Bearer k"}) as gw:
        gw.post("/sessions", json={"session_id": "s1"})
        r = gw.post("/sessions/s1/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 502
