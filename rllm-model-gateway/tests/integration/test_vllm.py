"""Integration tests against a real vLLM server on localhost:4000.

These tests require a running vLLM server. They are skipped automatically
if the server is not reachable.  Run with::

    pytest tests/integration/ -v
"""

import httpx
import pytest
import pytest_asyncio
from rllm_model_gateway import GatewayConfig, create_app

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

VLLM_BASE = "http://localhost:4000"


def _vllm_reachable() -> bool:
    try:
        r = httpx.get(f"{VLLM_BASE}/v1/models", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _get_model_id() -> str:
    """Return the first model ID from the vLLM server."""
    r = httpx.get(f"{VLLM_BASE}/v1/models", timeout=3.0)
    return r.json()["data"][0]["id"]


requires_vllm = pytest.mark.skipif(
    not _vllm_reachable(),
    reason="vLLM server not reachable on localhost:4000",
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_id():
    return _get_model_id()


@pytest.fixture
def gateway_app():
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{VLLM_BASE}/v1", "worker_id": "vllm-0"}],
        health_check_interval=999,
        sync_traces=True,
    )
    return create_app(config)


@pytest_asyncio.fixture
async def client(gateway_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app),
        base_url="http://testserver",
    ) as c:
        yield c


# ------------------------------------------------------------------
# Non-streaming tests
# ------------------------------------------------------------------


@requires_vllm
class TestNonStreaming:
    @pytest.mark.asyncio
    async def test_chat_completion(self, client: httpx.AsyncClient, model_id: str):
        """Basic non-streaming chat completion through the gateway."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 20,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Should have valid completion
        assert "choices" in data
        assert len(data["choices"]) >= 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["choices"][0]["finish_reason"] in ("stop", "length")

        # vLLM fields should be stripped
        assert "prompt_token_ids" not in data
        assert "prompt_logprobs" not in data
        assert "kv_transfer_params" not in data
        choice = data["choices"][0]
        assert "token_ids" not in choice
        assert "stop_reason" not in choice

        # Standard OpenAI fields should be present
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_models_endpoint(self, client: httpx.AsyncClient, model_id: str):
        """GET /v1/models should proxy to vLLM."""
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        model_ids = [m["id"] for m in data["data"]]
        assert model_id in model_ids

    @pytest.mark.asyncio
    async def test_logprobs_injected(self, client: httpx.AsyncClient, model_id: str):
        """Gateway should inject logprobs=True so vLLM returns logprob data."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        # Even though the client didn't request logprobs, the gateway
        # should have injected it.  The response still has them (strip_vllm
        # only removes token_ids, not standard logprobs).
        data = resp.json()
        choice = data["choices"][0]
        assert "logprobs" in choice
        assert choice["logprobs"] is not None
        assert len(choice["logprobs"]["content"]) > 0


# ------------------------------------------------------------------
# Streaming tests
# ------------------------------------------------------------------


@requires_vllm
class TestStreaming:
    @pytest.mark.asyncio
    async def test_streaming_chat_completion(self, client: httpx.AsyncClient, model_id: str):
        """Streaming SSE responses should be forwarded correctly."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 20,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        text = resp.text
        lines = [line for line in text.strip().split("\n") if line.startswith("data: ")]
        assert len(lines) >= 2  # At least one content chunk + [DONE]
        assert lines[-1].strip() == "data: [DONE]"

        # Parse a content chunk
        import json

        content_parts = []
        for line in lines:
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue
            chunk = json.loads(data_str)
            assert "choices" in chunk
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])

        full_content = "".join(content_parts)
        assert len(full_content) > 0


# ------------------------------------------------------------------
# Session + trace capture
# ------------------------------------------------------------------


@requires_vllm
class TestTraceCapture:
    @pytest.mark.asyncio
    async def test_trace_captured_with_session(self, client: httpx.AsyncClient, model_id: str):
        """Proxied call with session ID should persist a trace with token data."""
        resp = await client.post(
            "/sessions/integ-test/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

        # Retrieve traces
        traces_resp = await client.get("/sessions/integ-test/traces")
        assert traces_resp.status_code == 200
        traces = traces_resp.json()
        assert len(traces) == 1

        trace = traces[0]
        assert trace["session_id"] == "integ-test"
        assert trace["model"] == model_id

        # Token IDs should be captured (vLLM with return_token_ids=True)
        assert len(trace["prompt_token_ids"]) > 0
        assert len(trace["completion_token_ids"]) > 0

        # Logprobs should be captured
        assert trace["logprobs"] is not None
        assert len(trace["logprobs"]) > 0

        # Messages and response should be captured
        assert len(trace["messages"]) >= 1
        assert trace["response_message"]["role"] == "assistant"
        assert len(trace["response_message"]["content"]) > 0

        assert trace["finish_reason"] in ("stop", "length")
        assert trace["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_trace_captured_streaming(self, client: httpx.AsyncClient, model_id: str):
        """Streaming call with session should also capture a complete trace."""
        resp = await client.post(
            "/sessions/integ-stream/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 10,
                "stream": True,
            },
        )
        assert resp.status_code == 200

        # Consume the full streaming response
        _ = resp.text

        # Retrieve traces
        traces_resp = await client.get("/sessions/integ-stream/traces")
        assert traces_resp.status_code == 200
        traces = traces_resp.json()
        assert len(traces) == 1

        trace = traces[0]
        assert trace["session_id"] == "integ-stream"

        # Token IDs from streaming chunks
        assert len(trace["prompt_token_ids"]) > 0
        assert len(trace["completion_token_ids"]) > 0

        # Logprobs from streaming chunks
        assert trace["logprobs"] is not None
        assert len(trace["logprobs"]) > 0

        # Assembled response message
        assert trace["response_message"]["role"] == "assistant"
        assert len(trace["response_message"].get("content", "")) > 0

        assert trace["finish_reason"] in ("stop", "length")

    @pytest.mark.asyncio
    async def test_multi_turn_session(self, client: httpx.AsyncClient, model_id: str):
        """Multiple calls with the same session should produce multiple traces."""
        for i in range(3):
            resp = await client.post(
                "/sessions/multi-turn/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": f"Count: {i}"}],
                    "max_tokens": 5,
                },
            )
            assert resp.status_code == 200

        traces_resp = await client.get("/sessions/multi-turn/traces")
        assert traces_resp.status_code == 200
        traces = traces_resp.json()
        assert len(traces) == 3

        # All traces should have the same session_id
        for trace in traces:
            assert trace["session_id"] == "multi-turn"
            assert len(trace["prompt_token_ids"]) > 0
            assert len(trace["completion_token_ids"]) > 0

    @pytest.mark.asyncio
    async def test_trace_retrieval_by_id(self, client: httpx.AsyncClient, model_id: str):
        """Individual trace retrieval by trace_id should work."""
        await client.post(
            "/sessions/by-id-test/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            },
        )

        traces = (await client.get("/sessions/by-id-test/traces")).json()
        trace_id = traces[0]["trace_id"]

        resp = await client.get(f"/traces/{trace_id}")
        assert resp.status_code == 200
        assert resp.json()["trace_id"] == trace_id
        assert resp.json()["session_id"] == "by-id-test"
