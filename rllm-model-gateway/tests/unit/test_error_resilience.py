"""Error resilience tests: worker failure, partial rollouts, cleanup.

Uses ControllableMockVLLMServer for failure injection.
"""

import httpx
import pytest
import pytest_asyncio
from rllm_model_gateway import GatewayConfig, create_app

from tests.helpers.mock_vllm import ControllableMockVLLMServer


@pytest.fixture
def controllable_vllm():
    server = ControllableMockVLLMServer(port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture(params=["memory", "sqlite"])
def gateway_app(request, controllable_vllm: ControllableMockVLLMServer, tmp_path):
    store_worker = request.param
    config_kwargs = {
        "store_worker": store_worker,
        "workers": [{"url": f"{controllable_vllm.url}/v1", "worker_id": "w0"}],
        "health_check_interval": 999,
        "sync_traces": True,
    }
    if store_worker == "sqlite":
        config_kwargs["db_path"] = str(tmp_path / "test_traces.db")
    config = GatewayConfig(**config_kwargs)
    return create_app(config)


@pytest_asyncio.fixture
async def client(gateway_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app),
        base_url="http://testserver",
    ) as c:
        yield c


class TestWorkerFailure:
    @pytest.mark.asyncio
    async def test_worker_500_proxied(
        self,
        client: httpx.AsyncClient,
        controllable_vllm: ControllableMockVLLMServer,
    ):
        """Mock returns 500 -> gateway forwards the 500 status code."""
        controllable_vllm.set_fail(True)
        resp = await client.post(
            "/sessions/fail-test/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 500

        # A trace may still be stored (with empty token data) since the
        # gateway records the raw request/response pair regardless of status.
        traces_resp = await client.get("/sessions/fail-test/traces")
        traces = traces_resp.json()
        if len(traces) > 0:
            assert traces[0]["completion_token_ids"] == []

    @pytest.mark.asyncio
    async def test_malformed_response(
        self,
        client: httpx.AsyncClient,
        controllable_vllm: ControllableMockVLLMServer,
    ):
        """Mock returns invalid JSON -> gateway returns error gracefully."""
        controllable_vllm.set_malform(True)
        resp = await client.post(
            "/sessions/malformed-test/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        # Gateway should still return a response (proxied as-is or error)
        # The key is it doesn't crash
        assert resp.status_code in (200, 500, 502)

    @pytest.mark.asyncio
    async def test_partial_rollout_recovery(
        self,
        client: httpx.AsyncClient,
        controllable_vllm: ControllableMockVLLMServer,
    ):
        """2 successful calls then 500 -> 2 traces preserved (critical for RL)."""
        # Two successful calls
        for i in range(2):
            resp = await client.post(
                "/sessions/partial/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": f"msg {i}"}],
                },
            )
            assert resp.status_code == 200

        # Now fail
        controllable_vllm.set_fail(True)
        resp = await client.post(
            "/sessions/partial/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "msg 2"}],
            },
        )
        assert resp.status_code == 500

        # The 2 successful traces should still be there (a 3rd error trace may also exist)
        traces_resp = await client.get("/sessions/partial/traces")
        traces = traces_resp.json()
        assert len(traces) >= 2
        successful_traces = [t for t in traces if len(t["completion_token_ids"]) > 0]
        assert len(successful_traces) == 2
        for trace in successful_traces:
            assert trace["session_id"] == "partial"
            assert len(trace["prompt_token_ids"]) > 0

    @pytest.mark.asyncio
    async def test_no_healthy_workers(self, client: httpx.AsyncClient):
        """All workers removed -> error raised (RuntimeError or HTTP 5xx)."""
        # Remove the only worker
        workers_resp = await client.get("/admin/workers")
        workers = workers_resp.json()
        for w in workers:
            await client.delete(f"/admin/workers/{w['worker_id']}")

        # With no workers, the router raises RuntimeError which may propagate
        # as an unhandled exception through ASGI or be caught as HTTP 500.
        try:
            resp = await client.post(
                "/sessions/no-workers/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            # If the gateway handles the error gracefully
            assert resp.status_code in (500, 502, 503)
        except RuntimeError as e:
            # Expected: router raises when no workers available
            assert "No healthy workers" in str(e)


class TestSessionCleanup:
    @pytest.mark.asyncio
    async def test_delete_cleans_traces(self, client: httpx.AsyncClient):
        """Delete session -> traces gone."""
        # Create session and make a call
        await client.post("/sessions", json={"session_id": "cleanup-test"})
        resp = await client.post(
            "/sessions/cleanup-test/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200

        # Verify trace exists
        traces_resp = await client.get("/sessions/cleanup-test/traces")
        assert len(traces_resp.json()) == 1

        # Delete session
        await client.delete("/sessions/cleanup-test")

        # Traces should be gone (or session not found)
        traces_resp = await client.get("/sessions/cleanup-test/traces")
        traces = traces_resp.json()
        assert len(traces) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, client: httpx.AsyncClient):
        """Delete unknown session -> no crash."""
        resp = await client.delete("/sessions/nonexistent-session")
        # Should not crash; may return 200 with deleted=0 or 404
        assert resp.status_code in (200, 404)
