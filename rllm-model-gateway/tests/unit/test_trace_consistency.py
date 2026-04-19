"""Trace shape parity: passthrough mode and adapter mode produce structurally
identical TraceRecords for the same inbound request.
"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from rllm_model_gateway import (
    GatewayConfig,
    NormalizedRequest,
    NormalizedResponse,
    Usage,
    create_app,
)
from rllm_model_gateway.store.memory_store import MemoryTraceStore


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def fake_upstream():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "from upstream"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
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


def test_passthrough_and_adapter_traces_have_same_shape(fake_upstream):
    port = fake_upstream

    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        return NormalizedResponse(
            content="from adapter",
            finish_reason="stop",
            usage=Usage(prompt_tokens=5, completion_tokens=3),
        )

    same_request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    # Passthrough mode (upstream_url ends in /v1, OpenAI convention)
    pt_config = GatewayConfig(
        upstream_url=f"http://127.0.0.1:{port}/v1",
        admin_api_key="k",
        agent_api_key="k",
    )
    pt_app = create_app(pt_config, store=MemoryTraceStore())

    # Adapter mode
    ad_config = GatewayConfig(admin_api_key="k", agent_api_key="k")
    ad_app = create_app(ad_config, adapter=adapter, store=MemoryTraceStore())

    auth = {"Authorization": "Bearer k"}
    with TestClient(pt_app, headers=auth) as pt:
        pt.post("/sessions", json={"session_id": "s_pt"})
        r1 = pt.post("/sessions/s_pt/v1/chat/completions", json=same_request)
        assert r1.status_code == 200, r1.text
        pt.post("/admin/flush")
        traces_pt = pt.get("/sessions/s_pt/traces").json()

    with TestClient(ad_app, headers=auth) as ad:
        ad.post("/sessions", json={"session_id": "s_ad"})
        r2 = ad.post("/sessions/s_ad/v1/chat/completions", json=same_request)
        assert r2.status_code == 200
        ad.post("/admin/flush")
        traces_ad = ad.get("/sessions/s_ad/traces").json()

    assert len(traces_pt) == 1 and len(traces_ad) == 1
    t1, t2 = traces_pt[0], traces_ad[0]

    for key in ("messages", "content", "finish_reason", "endpoint", "model", "tools", "kwargs", "tool_calls", "reasoning"):
        assert key in t1, f"passthrough trace missing {key}"
        assert key in t2, f"adapter trace missing {key}"

    assert t1["messages"] == t2["messages"]
    assert t1["endpoint"] == t2["endpoint"] == "chat_completions"
