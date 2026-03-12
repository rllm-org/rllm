"""Concurrent session tests simulating batch RL rollouts.

Uses httpx.AsyncClient + ASGITransport (no real server) for speed.
"""

import asyncio

import httpx
import pytest
import pytest_asyncio
from rllm_model_gateway import GatewayConfig, create_app

from tests.helpers.mock_vllm import MockVLLMServer


@pytest.fixture(params=["memory", "sqlite"])
def gateway_app(request, mock_vllm: MockVLLMServer, tmp_path):
    store_worker = request.param
    config_kwargs = {
        "store_worker": store_worker,
        "workers": [{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
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


async def _do_chat(client: httpx.AsyncClient, session_id: str, stream: bool = False) -> int:
    """Make a chat completion request and return the status code."""
    resp = await client.post(
        f"/sessions/{session_id}/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": f"msg for {session_id}"}],
            "stream": stream,
        },
    )
    return resp.status_code


class TestSessionIsolation:
    @pytest.mark.asyncio
    async def test_session_isolation(self, client: httpx.AsyncClient):
        """10 sessions, 1 request each via asyncio.gather -> each has exactly 1 trace."""
        n = 10
        session_ids = [f"iso-{i}" for i in range(n)]

        statuses = await asyncio.gather(*[_do_chat(client, sid) for sid in session_ids])
        assert all(s == 200 for s in statuses)

        for sid in session_ids:
            resp = await client.get(f"/sessions/{sid}/traces")
            traces = resp.json()
            assert len(traces) == 1, f"Session {sid} has {len(traces)} traces, expected 1"
            assert traces[0]["session_id"] == sid

    @pytest.mark.asyncio
    async def test_concurrent_multi_turn(self, client: httpx.AsyncClient):
        """5 sessions, 3 turns each concurrently -> each has 3 traces."""
        n_sessions = 5
        n_turns = 3
        session_ids = [f"multi-{i}" for i in range(n_sessions)]

        async def multi_turn(sid: str):
            for _ in range(n_turns):
                status = await _do_chat(client, sid)
                assert status == 200

        await asyncio.gather(*[multi_turn(sid) for sid in session_ids])

        for sid in session_ids:
            resp = await client.get(f"/sessions/{sid}/traces")
            traces = resp.json()
            assert len(traces) == n_turns, f"Session {sid} has {len(traces)} traces, expected {n_turns}"
            for trace in traces:
                assert trace["session_id"] == sid

    @pytest.mark.asyncio
    async def test_high_concurrency(self, client: httpx.AsyncClient):
        """50 concurrent sessions -> no data corruption or lost traces."""
        n = 50
        session_ids = [f"high-{i}" for i in range(n)]

        statuses = await asyncio.gather(*[_do_chat(client, sid) for sid in session_ids])
        assert all(s == 200 for s in statuses)

        for sid in session_ids:
            resp = await client.get(f"/sessions/{sid}/traces")
            traces = resp.json()
            assert len(traces) == 1, f"Session {sid} has {len(traces)} traces, expected 1"
            assert traces[0]["session_id"] == sid
            assert len(traces[0]["prompt_token_ids"]) > 0
            assert len(traces[0]["completion_token_ids"]) > 0

    @pytest.mark.asyncio
    async def test_session_isolation_streaming(self, client: httpx.AsyncClient):
        """Same as isolation test with stream=True."""
        n = 10
        session_ids = [f"stream-iso-{i}" for i in range(n)]

        statuses = await asyncio.gather(*[_do_chat(client, sid, stream=True) for sid in session_ids])
        assert all(s == 200 for s in statuses)

        # Streaming traces are persisted via background tasks; poll until they appear
        for sid in session_ids:
            deadline = asyncio.get_event_loop().time() + 10.0
            traces = []
            while asyncio.get_event_loop().time() < deadline:
                resp = await client.get(f"/sessions/{sid}/traces")
                traces = resp.json()
                if len(traces) >= 1:
                    break
                await asyncio.sleep(0.1)
            assert len(traces) == 1, f"Session {sid} has {len(traces)} traces, expected 1"
            assert traces[0]["session_id"] == sid
