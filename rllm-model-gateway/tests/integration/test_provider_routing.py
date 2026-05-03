"""End-to-end test of the upstream-routing flow.

Stands up a fake OpenAI-compatible server alongside the gateway. The
gateway is configured with a single ``UpstreamRoute`` pointing at it.
A client posts to the gateway with ``X-RLLM-*`` headers; we then assert
the body forwards untouched, the auth header gets replaced with the
route's ``auth_header``, no rLLM scaffolding leaks, and the trace lands
with full session metadata.
"""

import json
import socket
import threading
import time
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from rllm_model_gateway import GatewayConfig, UpstreamRoute, create_app

from tests.helpers.gateway_server import GatewayServer

_CANNED_RESPONSE = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "model": "gpt-5-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello from fake provider"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
}


def _stream_chunks():
    """Yield SSE chunks shaped like an OpenAI streaming response."""
    chunks = [
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "model": "gpt-5-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "model": "gpt-5-mini",
            "choices": [{"index": 0, "delta": {"content": "hello from fake provider"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "model": "gpt-5-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _build_fake_provider_app() -> FastAPI:
    app = FastAPI()
    app.state.requests: list[dict[str, Any]] = []
    app.state.headers: list[dict[str, str]] = []
    app.state._lock = threading.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        with app.state._lock:
            app.state.requests.append(body)
            app.state.headers.append(dict(request.headers))
        if body.get("stream"):
            return StreamingResponse(_stream_chunks(), media_type="text/event-stream")
        return JSONResponse(content=_CANNED_RESPONSE)

    return app


def _reserve_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class _FakeProvider:
    def __init__(self) -> None:
        self.host = "127.0.0.1"
        self.port = _reserve_port(self.host)
        self.app = _build_fake_provider_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.app.state.requests

    @property
    def headers(self) -> list[dict[str, str]]:
        return self.app.state.headers

    def start(self) -> None:
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("Fake provider failed to start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)


@pytest.fixture
def fake_provider():
    p = _FakeProvider()
    p.start()
    yield p
    p.stop()


@pytest.fixture
def gateway_with_upstream(fake_provider, tmp_path):
    config = GatewayConfig(
        store_worker="sqlite",
        db_path=str(tmp_path / "traces.db"),
        sync_traces=True,
        # Disable vLLM-specific injections — eval-style configuration.
        add_logprobs=False,
        add_return_token_ids=False,
        strip_vllm_fields=False,
        routes={
            "gpt-5-mini": UpstreamRoute(
                upstream_url=f"{fake_provider.url}/v1",
                auth_header="Bearer sk-fake-test",
            ),
        },
    )
    app = create_app(config)
    server = GatewayServer(app, port=0)
    server.start()
    yield server, fake_provider
    server.stop()


def test_eval_flow_forwards_body_replaces_auth_and_persists_trace(gateway_with_upstream):
    """Full upstream flow: client → gateway → upstream, with header-stamped metadata."""
    gateway, provider = gateway_with_upstream

    resp = httpx.post(
        f"{gateway.url}/v1/chat/completions",
        json={
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={
            "X-RLLM-Session-Id": "sess-abc",
            "X-RLLM-Run-Id": "run-1",
            "X-RLLM-Harness": "opencode",
            "X-RLLM-Step-Id": "2",
            "Authorization": "Bearer client-original-key",
            "x-api-key": "client-supplied-anthropic-key",
        },
        timeout=10.0,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello from fake provider"

    # Upstream sees the body unchanged: model is forwarded as-sent, not rewritten.
    assert len(provider.requests) == 1
    fwd_body = provider.requests[0]
    assert fwd_body["model"] == "gpt-5-mini"
    assert fwd_body["messages"] == [{"role": "user", "content": "hi"}]

    fwd_headers = provider.headers[0]
    # Client's Authorization is replaced with the route's auth_header.
    assert fwd_headers.get("authorization") == "Bearer sk-fake-test"
    # x-api-key from the client never reaches the upstream.
    assert "x-api-key" not in fwd_headers
    # X-RLLM-* headers stay gateway-internal.
    for h in (
        "x-rllm-session-id",
        "x-rllm-run-id",
        "x-rllm-harness",
        "x-rllm-step-id",
    ):
        assert h not in fwd_headers, f"gateway leaked {h} to upstream"

    # Trace persisted with full metadata.
    traces_resp = httpx.get(f"{gateway.url}/sessions/sess-abc/traces", timeout=5.0)
    assert traces_resp.status_code == 200
    traces = traces_resp.json()
    assert len(traces) == 1
    trace = traces[0]
    assert trace["session_id"] == "sess-abc"
    assert trace["run_id"] == "run-1"
    assert trace["harness"] == "opencode"
    assert trace["step_id"] == 2
    assert trace["span_type"] == "llm.call"
    assert trace["model"] == "gpt-5-mini"


def test_eval_flow_streaming(gateway_with_upstream):
    """Streaming variant: gateway proxies SSE chunks and still records the trace."""
    gateway, provider = gateway_with_upstream

    with httpx.stream(
        "POST",
        f"{gateway.url}/v1/chat/completions",
        json={
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={
            "X-RLLM-Session-Id": "sess-stream",
            "X-RLLM-Run-Id": "run-s",
            "X-RLLM-Harness": "react",
        },
        timeout=10.0,
    ) as resp:
        assert resp.status_code == 200
        chunks_seen: list[dict[str, Any]] = []
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                break
            chunks_seen.append(json.loads(payload))

    assert chunks_seen, "expected at least one SSE chunk"
    assert any(chunk.get("choices", [{}])[0].get("delta", {}).get("content") == "hello from fake provider" for chunk in chunks_seen)

    # Drain pending trace writes before reading.
    httpx.post(f"{gateway.url}/admin/flush", timeout=5.0)

    traces = httpx.get(f"{gateway.url}/sessions/sess-stream/traces", timeout=5.0).json()
    assert len(traces) == 1
    trace = traces[0]
    assert trace["session_id"] == "sess-stream"
    assert trace["harness"] == "react"
    assert trace["run_id"] == "run-s"


def test_body_metadata_fallback_when_no_headers(gateway_with_upstream):
    """Clients that can't set headers can stamp metadata via request body."""
    gateway, provider = gateway_with_upstream

    resp = httpx.post(
        f"{gateway.url}/v1/chat/completions",
        json={
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"rllm": {"session_id": "sess-body", "harness": "claude-code"}},
        },
        timeout=10.0,
    )
    assert resp.status_code == 200

    # Body-stamped metadata should not leak to the upstream.
    fwd_body = provider.requests[-1]
    assert "rllm" not in fwd_body
    assert "metadata" not in fwd_body  # whole metadata dict was empty after rllm strip

    traces = httpx.get(f"{gateway.url}/sessions/sess-body/traces", timeout=5.0).json()
    assert len(traces) == 1
    assert traces[0]["harness"] == "claude-code"


def test_drop_params_removes_listed_fields(fake_provider, tmp_path):
    """A route with drop_params strips those fields from the forwarded body."""
    config = GatewayConfig(
        store_worker="memory",
        db_path=str(tmp_path / "traces.db"),
        sync_traces=True,
        add_logprobs=False,
        add_return_token_ids=False,
        strip_vllm_fields=False,
        routes={
            "gpt-5-mini": UpstreamRoute(
                upstream_url=f"{fake_provider.url}/v1",
                auth_header="Bearer sk-fake-test",
                drop_params=["max_tokens"],
            ),
        },
    )
    app = create_app(config)
    server = GatewayServer(app, port=0)
    server.start()
    try:
        resp = httpx.post(
            f"{server.url}/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "temperature": 0.7,
            },
            headers={"X-RLLM-Session-Id": "sess-drop"},
            timeout=10.0,
        )
        assert resp.status_code == 200
        fwd_body = fake_provider.requests[-1]
        assert "max_tokens" not in fwd_body
        assert fwd_body.get("temperature") == 0.7
    finally:
        server.stop()
