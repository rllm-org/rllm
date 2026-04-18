"""Tests for audit-pass fixes:
- AdapterError surfaces custom status codes
- n>1 rejected in adapter mode
- Sync adapter rejected at create_app
- metrics + metadata round-trip via API
- Session metadata round-trip via API
- get_session_traces since/limit query params
- Auto-generated session_id
"""

from __future__ import annotations

import pytest
from rllm_model_gateway import (
    AdapterError,
    GatewayConfig,
    NormalizedRequest,
    NormalizedResponse,
    Usage,
    create_app,
)
from rllm_model_gateway.store.memory_store import MemoryTraceStore

# ---------------------------------------------------------------------------
# AdapterError
# ---------------------------------------------------------------------------


def test_adapter_error_returns_custom_status(gateway_app_factory):
    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        raise AdapterError("rate limited by upstream", status_code=429)

    with gateway_app_factory(adapter=adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 429
        assert r.json()["error"] == "rate limited by upstream"


def test_adapter_error_400(gateway_app_factory):
    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        raise AdapterError("model rejected input", status_code=400)

    with gateway_app_factory(adapter=adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# n>1 rejection in adapter mode
# ---------------------------------------------------------------------------


def test_n_greater_than_one_rejected_in_adapter_mode(gateway_app_factory):
    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        return NormalizedResponse(content="ok", finish_reason="stop")

    with gateway_app_factory(adapter=adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "n": 3,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 400
        assert "n>1" in r.json()["error"]


def test_n_equals_one_accepted(gateway_app_factory):
    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        return NormalizedResponse(content="ok", finish_reason="stop")

    with gateway_app_factory(adapter=adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "n": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Sync adapter rejected at create_app
# ---------------------------------------------------------------------------


def test_sync_adapter_rejected():
    def sync_adapter(req):  # forgot async — should fail loudly
        return NormalizedResponse(content="ok", finish_reason="stop")

    cfg = GatewayConfig(admin_api_key="k", agent_api_key="k")
    with pytest.raises(TypeError, match="async def"):
        create_app(cfg, adapter=sync_adapter, store=MemoryTraceStore())


# ---------------------------------------------------------------------------
# metrics + metadata round-trip via API
# ---------------------------------------------------------------------------


def test_metrics_and_metadata_persisted_and_returned(gateway_app_factory):
    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        return NormalizedResponse(
            content="ok",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            metrics={"queue_time_ms": 12.5, "kv_cache_hit_rate": 0.83},
            metadata={"worker_id": "vllm-3", "backend": "vllm"},
        )

    with gateway_app_factory(adapter=adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        client.post("/sessions/s1/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        client.post("/admin/flush")
        traces = client.get("/sessions/s1/traces").json()
        t = traces[0]
        # Adapter-emitted metrics + gateway-emitted gateway_latency_ms.
        assert t["metrics"]["queue_time_ms"] == 12.5
        assert t["metrics"]["kv_cache_hit_rate"] == 0.83
        assert "gateway_latency_ms" in t["metrics"]
        # Adapter-emitted metadata.
        assert t["metadata"]["worker_id"] == "vllm-3"
        assert t["metadata"]["backend"] == "vllm"


# ---------------------------------------------------------------------------
# Session metadata round-trip via API
# ---------------------------------------------------------------------------


def test_session_metadata_round_trips(gateway_app_factory):
    with gateway_app_factory() as client:
        client.post(
            "/sessions",
            json={
                "session_id": "s1",
                "metadata": {"task_id": "math-001", "split": "validation"},
            },
        )
        info = client.get("/sessions/s1").json()
        assert info["metadata"] == {"task_id": "math-001", "split": "validation"}

        # list_sessions returns the metadata too
        sessions = client.get("/sessions").json()
        s1 = next(s for s in sessions if s["session_id"] == "s1")
        assert s1["metadata"] == {"task_id": "math-001", "split": "validation"}


# ---------------------------------------------------------------------------
# get_session_traces since / limit query params
# ---------------------------------------------------------------------------


def test_get_session_traces_respects_limit(gateway_app_factory):
    async def adapter(req):
        return NormalizedResponse(content="ok", finish_reason="stop")

    with gateway_app_factory(adapter=adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        for _ in range(5):
            client.post("/sessions/s1/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        client.post("/admin/flush")

        all_traces = client.get("/sessions/s1/traces").json()
        assert len(all_traces) == 5

        limited = client.get("/sessions/s1/traces?limit=2").json()
        assert len(limited) == 2


def test_get_session_traces_respects_since(gateway_app_factory):
    async def adapter(req):
        return NormalizedResponse(content="ok", finish_reason="stop")

    with gateway_app_factory(adapter=adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        client.post("/sessions/s1/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        client.post("/admin/flush")
        traces = client.get("/sessions/s1/traces").json()
        first_ts = traces[0]["timestamp"]

        # Anything strictly after the existing trace should be empty.
        future = client.get(f"/sessions/s1/traces?since={first_ts + 1.0}").json()
        assert future == []

        # Anything at or after returns the existing trace.
        present = client.get(f"/sessions/s1/traces?since={first_ts}").json()
        assert len(present) == 1


# ---------------------------------------------------------------------------
# Auto-generated session_id
# ---------------------------------------------------------------------------


def test_create_session_auto_generates_id(gateway_app_factory):
    with gateway_app_factory() as client:
        r = client.post("/sessions", json={})
        assert r.status_code == 200
        body = r.json()
        # Looks like a UUID — 32 hex chars + 4 dashes.
        assert len(body["session_id"]) == 36
        assert body["session_id"].count("-") == 4
        assert body["url"].endswith(f"/sessions/{body['session_id']}/v1")
