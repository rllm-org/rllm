"""Two-tier auth tests: admin vs agent key, header conventions, public /health."""

from __future__ import annotations

from fastapi.testclient import TestClient
from rllm_model_gateway import (
    GatewayConfig,
    NormalizedRequest,
    NormalizedResponse,
    create_app,
)


async def _adapter(req: NormalizedRequest) -> NormalizedResponse:
    return NormalizedResponse(content="ok", finish_reason="stop")


def _build_app():
    from rllm_model_gateway.store.memory_store import MemoryTraceStore

    config = GatewayConfig(
        admin_api_key="admin-secret",
        agent_api_key="agent-secret",
    )
    return create_app(config, adapter=_adapter, store=MemoryTraceStore())


def test_health_is_public():
    with TestClient(_build_app()) as client:
        # No Authorization header.
        assert client.get("/health").status_code == 200


def test_management_requires_admin_key():
    with TestClient(_build_app()) as client:
        # No auth → 401
        r = client.post("/sessions", json={"session_id": "s1"})
        assert r.status_code == 401

        # Agent key → 401 on management
        r = client.post(
            "/sessions",
            json={"session_id": "s1"},
            headers={"Authorization": "Bearer agent-secret"},
        )
        assert r.status_code == 401

        # Admin key → 200
        r = client.post(
            "/sessions",
            json={"session_id": "s1"},
            headers={"Authorization": "Bearer admin-secret"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == "s1"
        # agent_api_key is NOT returned per session (lives on GatewayConfig).
        assert "agent_api_key" not in body


def test_proxy_accepts_either_key():
    with TestClient(_build_app(), headers={"Authorization": "Bearer admin-secret"}) as client:
        client.post("/sessions", json={"session_id": "s1"})

        # Admin key → 200
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer admin-secret"},
        )
        assert r.status_code == 200

        # Agent key → 200
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer agent-secret"},
        )
        assert r.status_code == 200

        # No auth → 401
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": ""},
        )
        assert r.status_code == 401

        # Wrong key → 401
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer nope"},
        )
        assert r.status_code == 401


def test_x_api_key_header_accepted():
    """Anthropic-style x-api-key header works alongside Authorization: Bearer."""
    with TestClient(_build_app(), headers={"Authorization": "Bearer admin-secret"}) as client:
        client.post("/sessions", json={"session_id": "s1"})

        r = client.post(
            "/sessions/s1/v1/messages",
            json={"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "agent-secret", "Authorization": ""},
        )
        assert r.status_code == 200


def test_auto_generated_keys_when_not_provided():
    from rllm_model_gateway.store.memory_store import MemoryTraceStore

    config = GatewayConfig()
    create_app(config, adapter=_adapter, store=MemoryTraceStore())
    assert config.admin_api_key and len(config.admin_api_key) >= 32
    assert config.agent_api_key and len(config.agent_api_key) >= 32
    assert config.admin_api_key != config.agent_api_key
