"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from rllm_model_gateway import (
    GatewayConfig,
    NormalizedRequest,
    NormalizedResponse,
    ToolCall,
    Usage,
    create_app,
)
from rllm_model_gateway.store.memory_store import MemoryTraceStore


def make_fake_adapter(
    *,
    content: str = "Hello",
    reasoning: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
    usage: Usage | None = None,
    extras: dict | None = None,
    metrics: dict | None = None,
    metadata: dict | None = None,
):
    """Build an adapter fn returning a fixed NormalizedResponse."""

    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        return NormalizedResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls or [],
            finish_reason=finish_reason,
            usage=usage or Usage(prompt_tokens=10, completion_tokens=5),
            extras=extras or {},
            metrics=metrics or {},
            metadata=metadata or {},
        )

    return adapter


@pytest.fixture
def gateway_app_factory():
    """Factory: build a TestClient over an adapter-mode gateway."""

    def _factory(
        adapter=None,
        *,
        model: str | None = None,
        sampling_params_priority: str = "client",
    ):
        if adapter is None:
            adapter = make_fake_adapter()
        config = GatewayConfig(
            model=model,
            sampling_params_priority=sampling_params_priority,
            admin_api_key="test-admin-key",
            agent_api_key="test-agent-key",
        )
        app = create_app(config, adapter=adapter, store=MemoryTraceStore())
        return TestClient(app, headers={"Authorization": "Bearer test-admin-key"})

    return _factory


@pytest.fixture
def fake_adapter():
    return make_fake_adapter
