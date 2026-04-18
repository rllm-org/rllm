"""Integration test helpers: spin up a real uvicorn gateway on a free port."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from rllm_model_gateway import (
    GatewayConfig,
    NormalizedRequest,
    NormalizedResponse,
    ToolCall,
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


@pytest.fixture
def fake_adapter():
    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        return NormalizedResponse(
            content="Hello from adapter",
            reasoning="I considered options",
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"q": "rllm"}, arguments_raw='{"q":"rllm"}')],
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=12, completion_tokens=7),
            extras={"completion_ids": [10, 20, 30, 40, 50, 60, 70], "logprobs": [-0.1] * 7},
        )

    return adapter


@pytest.fixture
def gateway_server(fake_adapter):
    """Start a uvicorn gateway in adapter mode on a free port.

    Yields ``(base_url, admin_key, agent_key)``.
    """
    port = _free_port()
    config = GatewayConfig(
        host="127.0.0.1",
        port=port,
        admin_api_key="test-admin",
        agent_api_key="test-agent",
    )
    app = create_app(config, adapter=fake_adapter, store=MemoryTraceStore())

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    else:
        raise RuntimeError("gateway did not become healthy in time")

    yield base_url, config.admin_api_key, config.agent_api_key
    server.should_exit = True
    t.join(timeout=2.0)
