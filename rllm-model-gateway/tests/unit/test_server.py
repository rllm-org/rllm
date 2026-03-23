"""Integration tests for the gateway server (FastAPI app).

Uses httpx.AsyncClient as a test client against the app, with a real
mock vLLM backend server for proxy tests.
"""

import json

import httpx
import pytest
import pytest_asyncio
from rllm_model_gateway import GatewayConfig, create_app
from rllm_model_gateway.models import WorkerConfig, WorkerInfo, _split_worker_url
from rllm_model_gateway.proxy import ReverseProxy

from tests.helpers.mock_vllm import MockVLLMServer

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def app(mock_vllm: MockVLLMServer):
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,
    )
    return create_app(config)


@pytest_asyncio.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


# ------------------------------------------------------------------
# Health endpoints
# ------------------------------------------------------------------


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_workers(self, client: httpx.AsyncClient):
        resp = await client.get("/health/workers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1


# ------------------------------------------------------------------
# Session management
# ------------------------------------------------------------------


class TestSessions:
    @pytest.mark.asyncio
    async def test_create_session(self, client: httpx.AsyncClient):
        resp = await client.post("/sessions", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "url" in data

    @pytest.mark.asyncio
    async def test_create_session_with_id(self, client: httpx.AsyncClient):
        resp = await client.post("/sessions", json={"session_id": "my-session"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "my-session"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client: httpx.AsyncClient):
        resp = await client.get("/sessions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client: httpx.AsyncClient):
        resp = await client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_delete_session(self, client: httpx.AsyncClient):
        await client.post("/sessions", json={"session_id": "to-delete"})
        resp = await client.delete("/sessions/to-delete")
        assert resp.status_code == 200


# ------------------------------------------------------------------
# Admin / worker management
# ------------------------------------------------------------------


class TestAdmin:
    @pytest.mark.asyncio
    async def test_list_workers(self, client: httpx.AsyncClient):
        resp = await client.get("/admin/workers")
        assert resp.status_code == 200
        workers = resp.json()
        assert len(workers) >= 1
        assert workers[0]["worker_id"] == "w0"

    @pytest.mark.asyncio
    async def test_add_worker(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/admin/workers",
            json={"url": "http://new-worker:8000/v1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "worker_id" in data

        # Verify it shows up in list (URL is auto-split: base URL without /v1)
        workers = await client.get("/admin/workers")
        urls = [w["url"] for w in workers.json()]
        assert "http://new-worker:8000" in urls

    @pytest.mark.asyncio
    async def test_add_worker_missing_url(self, client: httpx.AsyncClient):
        resp = await client.post("/admin/workers", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_remove_worker(self, client: httpx.AsyncClient):
        add_resp = await client.post(
            "/admin/workers",
            json={"url": "http://temp:8000"},
        )
        wid = add_resp.json()["worker_id"]
        resp = await client.delete(f"/admin/workers/{wid}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_remove_nonexistent_worker(self, client: httpx.AsyncClient):
        resp = await client.delete("/admin/workers/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_flush(self, client: httpx.AsyncClient):
        resp = await client.post("/admin/flush")
        assert resp.status_code == 200
        assert resp.json()["status"] == "flushed"


# ------------------------------------------------------------------
# Proxy (non-streaming)
# ------------------------------------------------------------------


class TestProxy:
    @pytest.mark.asyncio
    async def test_chat_completions_bare(self, client: httpx.AsyncClient):
        """POST /v1/chat/completions without session — should proxy successfully."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        # vLLM fields should be stripped
        assert "prompt_token_ids" not in data
        assert "prompt_logprobs" not in data
        choice = data.get("choices", [{}])[0]
        assert "token_ids" not in choice
        assert "stop_reason" not in choice

    @pytest.mark.asyncio
    async def test_chat_completions_with_session(self, client: httpx.AsyncClient):
        """POST /sessions/{sid}/v1/chat/completions — should proxy and capture trace."""
        resp = await client.post(
            "/sessions/test-sess/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from mock!"

    @pytest.mark.asyncio
    async def test_models_endpoint(self, client: httpx.AsyncClient):
        """GET /v1/models should proxy to the worker."""
        resp = await client.get("/v1/models")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_logprobs_injected(self, client: httpx.AsyncClient, mock_vllm: MockVLLMServer):
        """Middleware should inject logprobs=True into the request body."""
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert len(mock_vllm.request_log) >= 1
        last_req = mock_vllm.request_log[-1]
        assert last_req.get("logprobs") is True
        assert last_req.get("return_token_ids") is True

    @pytest.mark.asyncio
    async def test_logprobs_stripped_when_not_requested(self, client: httpx.AsyncClient):
        """Response should NOT contain logprobs when client didn't ask for them."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        for choice in data.get("choices", []):
            assert "logprobs" not in choice, "logprobs leaked to client that didn't request them"

    @pytest.mark.asyncio
    async def test_logprobs_preserved_when_requested(self, client: httpx.AsyncClient):
        """Response should contain logprobs when client explicitly asked for them."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]
        assert "logprobs" in choice, "logprobs should be present when client requested them"
        assert choice["logprobs"]["content"] is not None


# ------------------------------------------------------------------
# Proxy (streaming)
# ------------------------------------------------------------------


class TestStreamingProxy:
    @pytest.mark.asyncio
    async def test_streaming_strips_vllm_fields(self, client: httpx.AsyncClient):
        """SSE chunks returned to agent must NOT contain vLLM-specific fields."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        lines = [line for line in resp.text.strip().split("\n") if line.startswith("data: ") and line.strip() != "data: [DONE]"]
        assert len(lines) >= 1

        for line in lines:
            chunk = json.loads(line[6:])
            # Root-level vLLM fields must be stripped
            assert "prompt_token_ids" not in chunk
            assert "prompt_logprobs" not in chunk
            assert "kv_transfer_params" not in chunk
            # Choice-level vLLM fields must be stripped
            for choice in chunk.get("choices", []):
                assert "token_ids" not in choice
                assert "stop_reason" not in choice

    @pytest.mark.asyncio
    async def test_streaming_content_intact(self, client: httpx.AsyncClient):
        """SSE streaming should deliver all content from the mock."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        content_parts = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", []):
                    c = choice.get("delta", {}).get("content")
                    if c:
                        content_parts.append(c)
        assert "".join(content_parts) == "Hello from mock!"

    @pytest.mark.asyncio
    async def test_streaming_logprobs_stripped_when_not_requested(self, client: httpx.AsyncClient):
        """SSE chunks should NOT contain logprobs when client didn't ask for them."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        for line in resp.text.strip().split("\n"):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", []):
                    assert "logprobs" not in choice, "logprobs leaked in SSE chunk"

    @pytest.mark.asyncio
    async def test_streaming_logprobs_preserved_when_requested(self, client: httpx.AsyncClient):
        """SSE chunks should contain logprobs when client explicitly asked for them."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "logprobs": True,
            },
        )
        assert resp.status_code == 200

        found_logprobs = False
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", []):
                    if choice.get("logprobs") is not None:
                        found_logprobs = True
        assert found_logprobs, "logprobs should be present in at least one SSE chunk when client requested them"

    @pytest.mark.asyncio
    async def test_streaming_trace_captured(self, mock_vllm: MockVLLMServer):
        """Streaming call with session should capture a trace with token data."""
        config = GatewayConfig(
            store_worker="memory",
            workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
            health_check_interval=999,
            sync_traces=True,
        )
        app = create_app(config)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/sessions/stream-test/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200

            traces_resp = await client.get("/sessions/stream-test/traces")
            traces = traces_resp.json()
            assert len(traces) == 1

            trace = traces[0]
            assert trace["session_id"] == "stream-test"
            assert trace["prompt_token_ids"] == [1, 2, 3, 4, 5]
            assert trace["completion_token_ids"] == [10, 11, 12]
            assert trace["logprobs"] == [-0.5, -0.3, -0.1]
            assert trace["response_message"]["role"] == "assistant"
            assert "Hello" in trace["response_message"]["content"]

    @pytest.mark.asyncio
    async def test_streaming_trace_intact_with_strip_vllm_off(self, mock_vllm: MockVLLMServer):
        """Logprobs stripping must not destroy trace data when strip_vllm=False.

        Regression test: when strip_vllm is disabled, the sanitized chunk
        aliases the original chunk in the trace buffer.  A mutating
        _strip_logprobs would destroy logprobs in the trace.
        """
        config = GatewayConfig(
            store_worker="memory",
            workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
            health_check_interval=999,
            sync_traces=True,
            strip_vllm_fields=False,
        )
        app = create_app(config)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            # Client does NOT request logprobs — gateway injects them,
            # then strips them from the response.
            resp = await client.post(
                "/sessions/no-strip/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200

            # Client-facing chunks should NOT have logprobs
            for line in resp.text.strip().split("\n"):
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    for choice in chunk.get("choices", []):
                        assert "logprobs" not in choice, "logprobs leaked to client"

            # Trace should still have logprobs intact
            traces = (await client.get("/sessions/no-strip/traces")).json()
            assert len(traces) == 1
            trace = traces[0]
            assert trace["logprobs"] == [-0.5, -0.3, -0.1], "logprobs in trace were destroyed by client-facing stripping"


# ------------------------------------------------------------------
# Trace capture and retrieval
# ------------------------------------------------------------------


class TestTraceCapture:
    @pytest.mark.asyncio
    async def test_trace_persisted_for_session(self, mock_vllm: MockVLLMServer):
        """After a proxied call with session, traces should be retrievable."""
        config = GatewayConfig(
            store_worker="memory",
            workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
            health_check_interval=999,
            sync_traces=True,
        )
        app = create_app(config)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/sessions/trace-test/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code == 200

            traces_resp = await client.get("/sessions/trace-test/traces")
            assert traces_resp.status_code == 200
            traces = traces_resp.json()
            assert len(traces) == 1

            trace = traces[0]
            assert trace["session_id"] == "trace-test"
            assert trace["prompt_token_ids"] == [1, 2, 3, 4, 5]
            assert trace["completion_token_ids"] == [10, 11, 12]
            assert trace["logprobs"] == [-0.5, -0.3, -0.1]
            assert trace["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_get_trace_by_id(self, mock_vllm: MockVLLMServer):
        config = GatewayConfig(
            store_worker="memory",
            workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
            health_check_interval=999,
            sync_traces=True,
        )
        app = create_app(config)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            await client.post(
                "/sessions/s1/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

            traces = (await client.get("/sessions/s1/traces")).json()
            trace_id = traces[0]["trace_id"]

            resp = await client.get(f"/traces/{trace_id}")
            assert resp.status_code == 200
            assert resp.json()["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_get_trace_not_found(self, client: httpx.AsyncClient):
        resp = await client.get("/traces/nonexistent")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# URL splitting and WorkerConfig auto-split
# ------------------------------------------------------------------


class TestUrlSplitting:
    def test_worker_url_split_with_path(self):
        result = _split_worker_url("http://localhost:4000/v1")
        assert result == {"url": "http://localhost:4000", "api_path": "/v1"}

    def test_worker_url_split_no_path(self):
        result = _split_worker_url("http://localhost:4000")
        assert result == {"url": "http://localhost:4000", "api_path": "/v1"}

    def test_worker_url_split_custom_path(self):
        result = _split_worker_url("http://localhost:4000/api/v2")
        assert result == {"url": "http://localhost:4000", "api_path": "/api/v2"}

    def test_worker_url_split_trailing_slash(self):
        result = _split_worker_url("http://localhost:4000/v1/")
        assert result == {"url": "http://localhost:4000", "api_path": "/v1"}


class TestWorkerConfigAutoSplit:
    def test_worker_config_auto_splits_url(self):
        wc = WorkerConfig(url="http://localhost:4000/v1")
        assert wc.url == "http://localhost:4000"
        assert wc.api_path == "/v1"

    def test_worker_config_explicit_api_path(self):
        wc = WorkerConfig(url="http://localhost:4000", api_path="/v2")
        assert wc.url == "http://localhost:4000"
        assert wc.api_path == "/v2"

    def test_worker_config_no_path_defaults(self):
        wc = WorkerConfig(url="http://localhost:4000")
        assert wc.url == "http://localhost:4000"
        assert wc.api_path == "/v1"

    def test_worker_config_dict_input(self):
        """Backward compat: dict with url containing path auto-splits."""
        wc = WorkerConfig(**{"url": "http://localhost:4000/v1", "worker_id": "w0"})
        assert wc.url == "http://localhost:4000"
        assert wc.api_path == "/v1"


class TestBuildUrl:
    """Test ReverseProxy._build_url with various api_path values."""

    def test_default_v1_api_path(self):
        url = ReverseProxy._build_url("http://host:4000/v1", "/v1/chat/completions", "")
        assert url == "http://host:4000/v1/chat/completions"

    def test_custom_api_path(self):
        """Custom api_path like /api/v2 should work correctly."""
        # Worker has api_path="/api/v2", so api_url is "http://host:4000/api/v2"
        # Gateway request path is always /v1/... which gets stripped to /chat/completions
        url = ReverseProxy._build_url("http://host:4000/api/v2", "/v1/chat/completions", "")
        assert url == "http://host:4000/api/v2/chat/completions"

    def test_with_query_string(self):
        url = ReverseProxy._build_url("http://host:4000/v1", "/v1/models", "limit=10")
        assert url == "http://host:4000/v1/models?limit=10"

    def test_bare_v1_path(self):
        url = ReverseProxy._build_url("http://host:4000/v1", "/v1", "")
        assert url == "http://host:4000/v1"

    def test_no_api_path_on_worker(self):
        url = ReverseProxy._build_url("http://host:4000", "/v1/chat/completions", "")
        assert url == "http://host:4000/chat/completions"


class TestWorkerInfoAutoSplit:
    """Test WorkerInfo._auto_split_url model validator."""

    def test_auto_splits_url_with_path(self):
        w = WorkerInfo(worker_id="w0", url="http://host:4000/v1")
        assert w.url == "http://host:4000"
        assert w.api_path == "/v1"

    def test_explicit_api_path_not_overridden(self):
        w = WorkerInfo(worker_id="w0", url="http://host:4000", api_path="/api/v2")
        assert w.url == "http://host:4000"
        assert w.api_path == "/api/v2"

    def test_no_path_defaults_to_v1(self):
        w = WorkerInfo(worker_id="w0", url="http://host:4000")
        assert w.api_path == "/v1"


class TestAdminWorkerAutoSplit:
    @pytest.mark.asyncio
    async def test_add_worker_auto_splits_url(self, client: httpx.AsyncClient):
        """Admin add_worker should auto-split URL with path into url + api_path."""
        resp = await client.post(
            "/admin/workers",
            json={"url": "http://new-worker:8000/v1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] == "http://new-worker:8000"
        assert data["api_path"] == "/v1"

        # Verify it shows up correctly in list
        workers = await client.get("/admin/workers")
        w = next(w for w in workers.json() if w["url"] == "http://new-worker:8000")
        assert w["api_path"] == "/v1"
