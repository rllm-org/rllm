"""Tests for GatewayClient and AsyncGatewayClient.

Uses a real gateway server (uvicorn in a thread) with mock backend.
"""

import openai
import pytest
from rllm_model_gateway import AsyncGatewayClient, GatewayClient, GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer
from tests.helpers.mock_vllm import MockVLLMServer


@pytest.fixture(params=["memory", "sqlite"])
def gateway_server(request, mock_vllm: MockVLLMServer, tmp_path):
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
    app = create_app(config)
    server = GatewayServer(app, port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def gw_url(gateway_server: GatewayServer) -> str:
    return gateway_server.url


class TestSyncGatewayClient:
    def test_create_and_get_session(self, gw_url: str):
        with GatewayClient(gw_url) as client:
            sid = client.create_session(session_id="test-session")
            assert sid == "test-session"
            info = client.get_session_info(sid)
            assert info["session_id"] == "test-session"

    def test_create_session_auto_id(self, gw_url: str):
        with GatewayClient(gw_url) as client:
            sid = client.create_session()
            assert sid  # Should be a non-empty UUID
            assert len(sid) > 0
            info = client.get_session_info(sid)
            assert info["session_id"] == sid

    def test_list_sessions(self, gw_url: str):
        with GatewayClient(gw_url) as client:
            for i in range(3):
                sid = client.create_session(session_id=f"list-{i}")
                # Make a call so the session has traces and appears in list
                oai = openai.OpenAI(base_url=client.get_session_url(sid), api_key="dummy")
                oai.chat.completions.create(
                    model="mock-model",
                    messages=[{"role": "user", "content": "hi"}],
                )
            sessions = client.list_sessions()
            session_ids = [s["session_id"] for s in sessions]
            for i in range(3):
                assert f"list-{i}" in session_ids

    def test_delete_session(self, gw_url: str):
        import httpx

        with GatewayClient(gw_url) as client:
            sid = client.create_session(session_id="to-delete")
            # Make a call to generate a trace
            oai = openai.OpenAI(base_url=client.get_session_url(sid), api_key="dummy")
            oai.chat.completions.create(
                model="mock-model",
                messages=[{"role": "user", "content": "hi"}],
            )
            traces_before = client.get_session_traces(sid)
            assert len(traces_before) == 1

            client.delete_session(sid)
            # Session should be gone
            with pytest.raises(httpx.HTTPStatusError):
                client.get_session_info(sid)

    def test_get_session_traces_with_limit(self, gw_url: str):
        with GatewayClient(gw_url) as client:
            sid = client.create_session(session_id="limit-test")
            session_url = client.get_session_url(sid)
            oai = openai.OpenAI(base_url=session_url, api_key="dummy")
            for _ in range(3):
                oai.chat.completions.create(
                    model="mock-model",
                    messages=[{"role": "user", "content": "hi"}],
                )
            all_traces = client.get_session_traces(sid)
            assert len(all_traces) == 3
            limited = client.get_session_traces(sid, limit=2)
            assert len(limited) == 2

    def test_get_trace_by_id(self, gw_url: str):
        with GatewayClient(gw_url) as client:
            sid = client.create_session(session_id="trace-by-id")
            oai = openai.OpenAI(base_url=client.get_session_url(sid), api_key="dummy")
            oai.chat.completions.create(
                model="mock-model",
                messages=[{"role": "user", "content": "hi"}],
            )
            traces = client.get_session_traces(sid)
            trace_id = traces[0].trace_id
            trace = client.get_trace(trace_id)
            assert trace.trace_id == trace_id
            assert trace.session_id == sid

    def test_add_and_remove_worker(self, gw_url: str):
        with GatewayClient(gw_url) as client:
            wid = client.add_worker(url="http://new-worker:8000/v1")
            workers = client.list_workers()
            worker_ids = [w.worker_id for w in workers]
            assert wid in worker_ids

            client.remove_worker(wid)
            workers = client.list_workers()
            worker_ids = [w.worker_id for w in workers]
            assert wid not in worker_ids

    def test_health_check(self, gw_url: str):
        with GatewayClient(gw_url) as client:
            health = client.health()
            assert health["status"] == "ok"

    def test_context_manager(self, gw_url: str):
        with GatewayClient(gw_url) as client:
            health = client.health()
            assert health["status"] == "ok"
        # After exiting context, the client should be closed
        # (no assertion needed, just ensure no exception)


class TestAsyncGatewayClient:
    @pytest.mark.asyncio
    async def test_async_create_and_get_session(self, gw_url: str):
        async with AsyncGatewayClient(gw_url) as client:
            sid = await client.create_session(session_id="async-test")
            assert sid == "async-test"
            info = await client.get_session_info(sid)
            assert info["session_id"] == "async-test"

    @pytest.mark.asyncio
    async def test_async_get_traces(self, gw_url: str):
        async with AsyncGatewayClient(gw_url) as client:
            sid = await client.create_session(session_id="async-traces")
            session_url = client.get_session_url(sid)
            # Use sync openai client for the call (simplicity)
            oai = openai.OpenAI(base_url=session_url, api_key="dummy")
            oai.chat.completions.create(
                model="mock-model",
                messages=[{"role": "user", "content": "hi"}],
            )
            traces = await client.get_session_traces(sid)
            assert len(traces) == 1
            assert traces[0].session_id == sid

    @pytest.mark.asyncio
    async def test_async_context_manager(self, gw_url: str):
        async with AsyncGatewayClient(gw_url) as client:
            health = await client.health()
            assert health["status"] == "ok"
