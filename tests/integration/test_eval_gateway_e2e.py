"""E2E test of EvalGatewayManager against a fake provider.

Patches the provider registry's ``backend_url`` to point at an in-process
fake "OpenAI" server, boots the gateway, and confirms a chat-completions
roundtrip works (rewritten model, injected auth, persisted trace).
"""

import socket
import threading
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rllm.eval.config import ProviderInfo
from rllm.eval.gateway import EvalGatewayManager

_CANNED_RESPONSE = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "model": "gpt-5-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _reserve_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


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
        return JSONResponse(content=_CANNED_RESPONSE)

    return app


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
def patched_openai_provider(fake_provider):
    """Override the OpenAI provider's backend_url to point at the fake."""
    fake_info = ProviderInfo(
        id="openai",
        label="OpenAI",
        litellm_prefix="openai",
        env_key="OPENAI_API_KEY",
        default_model="gpt-5-mini",
        models=["gpt-5-mini"],
        backend_url=f"{fake_provider.url}/v1",
    )
    with patch("rllm.eval.gateway.get_provider_info", return_value=fake_info):
        yield fake_provider


def test_eval_gateway_manager_full_roundtrip(patched_openai_provider, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gm = EvalGatewayManager(
        provider="openai",
        model_name="gpt-5-mini",
        api_key="sk-fake-test",
    )
    url = gm.start()
    try:
        resp = httpx.post(
            f"{url}/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "X-RLLM-Session-Id": "eval-0",
                "X-RLLM-Run-Id": "run-1",
                "X-RLLM-Harness": "react",
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "ok"

        # Provider received: rewritten body + provider-issued auth.
        assert len(patched_openai_provider.requests) == 1
        fwd_body = patched_openai_provider.requests[0]
        assert fwd_body["model"] == "gpt-5-mini"
        fwd_headers = patched_openai_provider.headers[0]
        assert fwd_headers.get("authorization") == "Bearer sk-fake-test"
        assert "x-rllm-session-id" not in fwd_headers

        # Trace persisted with full metadata.
        traces = httpx.get(f"{url[:-3]}/sessions/eval-0/traces", timeout=5.0).json()
        # Drain async writes if anything is pending.
        if not traces:
            httpx.post(f"{url[:-3]}/admin/flush", timeout=5.0)
            traces = httpx.get(f"{url[:-3]}/sessions/eval-0/traces", timeout=5.0).json()
        assert len(traces) == 1
        trace = traces[0]
        assert trace["session_id"] == "eval-0"
        assert trace["run_id"] == "run-1"
        assert trace["harness"] == "react"
    finally:
        gm.shutdown()


def test_eval_gateway_manager_double_start_is_idempotent(patched_openai_provider, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
    url1 = gm.start()
    try:
        url2 = gm.start()
        assert url1 == url2
    finally:
        gm.shutdown()


def test_shutdown_without_start_is_noop():
    gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
    gm.shutdown()  # should not raise
