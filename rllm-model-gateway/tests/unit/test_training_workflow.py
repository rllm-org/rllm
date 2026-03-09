"""Training workflow simulation tests using mock vLLM backend.

Tests the full training workflow: session creation -> OpenAI client calls ->
trace retrieval, all using a mock backend and real gateway server.
"""

import time

import litellm
import openai
import pytest
from rllm_model_gateway import GatewayClient, GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer
from tests.helpers.mock_vllm import MockVLLMServer


def _wait_for_traces(client: GatewayClient, session_id: str, expected: int, timeout: float = 10.0) -> list:
    """Poll until the expected number of traces appear (handles async trace writes)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        traces = client.get_session_traces(session_id)
        if len(traces) >= expected:
            return traces
        time.sleep(0.1)
    return client.get_session_traces(session_id)


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


class TestSingleRolloutLifecycle:
    def test_single_rollout_lifecycle(self, gw_url: str):
        """Full flow: create_session -> OpenAI chat -> get_session_traces -> verify fields."""

        client = GatewayClient(gw_url)
        sid = client.create_session(session_id="rollout-1")
        assert sid == "rollout-1"

        session_url = client.get_session_url(sid)
        oai = openai.OpenAI(base_url=session_url, api_key="dummy")
        resp = oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert resp.choices[0].message.content == "Hello from mock!"

        traces = client.get_session_traces(sid)
        assert len(traces) == 1
        trace = traces[0]
        assert trace.session_id == sid
        assert trace.prompt_token_ids == [1, 2, 3, 4, 5]
        assert trace.completion_token_ids == [10, 11, 12]
        assert trace.logprobs == [-0.5, -0.3, -0.1]
        assert trace.response_message["role"] == "assistant"
        assert "Hello" in trace.response_message["content"]
        assert trace.finish_reason == "stop"
        assert len(trace.messages) >= 1
        client.close()


class TestMultiTurnAgentConversation:
    def test_multi_turn_agent_conversation(self, gw_url: str):
        """3 sequential chat completions in one session -> 3 traces in order."""

        client = GatewayClient(gw_url)
        sid = client.create_session(session_id="multi-turn")
        session_url = client.get_session_url(sid)
        oai = openai.OpenAI(base_url=session_url, api_key="dummy")

        for i in range(3):
            oai.chat.completions.create(
                model="mock-model",
                messages=[{"role": "user", "content": f"Turn {i}"}],
            )

        traces = client.get_session_traces(sid)
        assert len(traces) == 3
        for trace in traces:
            assert trace.session_id == sid
            assert len(trace.prompt_token_ids) > 0
            assert len(trace.completion_token_ids) > 0
        client.close()


class TestStreamingRollout:
    def test_streaming_rollout(self, gw_url: str):
        """Streaming call -> verify trace has accumulated token IDs and logprobs."""

        client = GatewayClient(gw_url)
        sid = client.create_session(session_id="stream-rollout")
        session_url = client.get_session_url(sid)
        oai = openai.OpenAI(base_url=session_url, api_key="dummy")

        stream = oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        content_parts = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
        assert "".join(content_parts) == "Hello from mock!"

        # Streaming traces are written async; poll briefly
        traces = _wait_for_traces(client, sid, expected=1)
        assert len(traces) == 1
        trace = traces[0]
        assert trace.prompt_token_ids == [1, 2, 3, 4, 5]
        assert trace.completion_token_ids == [10, 11, 12]
        assert trace.logprobs == [-0.5, -0.3, -0.1]
        client.close()


class TestSessionWithMetadata:
    def test_session_with_metadata(self, gw_url: str):
        """Create session with metadata -> verify in get_session_info()."""
        client = GatewayClient(gw_url)
        sid = client.create_session(
            session_id="meta-session",
            metadata={"prompt_idx": 42},
        )
        info = client.get_session_info(sid)
        assert info["metadata"]["prompt_idx"] == 42
        client.close()


class TestRewardCorrelationPattern:
    def test_reward_correlation_pattern(self, gw_url: str):
        """Create N sessions, run rollouts, retrieve all traces, verify indexable by session_id."""

        client = GatewayClient(gw_url)
        session_ids = []
        for i in range(5):
            sid = client.create_session(session_id=f"reward-{i}")
            session_ids.append(sid)
            session_url = client.get_session_url(sid)
            oai = openai.OpenAI(base_url=session_url, api_key="dummy")
            oai.chat.completions.create(
                model="mock-model",
                messages=[{"role": "user", "content": f"Prompt {i}"}],
            )

        # Verify each session has exactly 1 trace
        traces_by_session = {}
        for sid in session_ids:
            traces = client.get_session_traces(sid)
            traces_by_session[sid] = traces
            assert len(traces) == 1
            assert traces[0].session_id == sid

        assert len(traces_by_session) == 5
        client.close()


class TestAgentZeroGatewayAwareness:
    def test_agent_zero_gateway_awareness(self, gw_url: str):
        """Standard openai.OpenAI with no rllm imports -> traces captured."""

        # Agent code: only uses openai, knows nothing about rllm
        oai = openai.OpenAI(
            base_url=f"{gw_url}/sessions/zero-aware/v1",
            api_key="dummy",
        )
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "test"}],
        )

        # Training side retrieves traces
        client = GatewayClient(gw_url)
        traces = client.get_session_traces("zero-aware")
        assert len(traces) == 1
        assert traces[0].session_id == "zero-aware"
        client.close()


class TestLiteLLMClientCompatibility:
    def test_litellm_client_compatibility(self, gw_url: str):
        """LiteLLM completion() with base_url pointing at gateway -> traces captured."""
        client = GatewayClient(gw_url)
        sid = client.create_session(session_id="litellm-test")
        session_url = client.get_session_url(sid)

        resp = litellm.completion(
            model="openai/mock-model",
            messages=[{"role": "user", "content": "Hello from litellm"}],
            api_base=session_url,
            api_key="dummy",
        )
        assert resp.choices[0].message.content == "Hello from mock!"

        traces = client.get_session_traces(sid)
        assert len(traces) == 1
        assert traces[0].session_id == sid
        client.close()


class TestImplicitSessionCreation:
    def test_implicit_session_creation(self, gw_url: str):
        """Hit /sessions/{new_sid}/v1/chat/completions without create_session() -> session created."""

        oai = openai.OpenAI(
            base_url=f"{gw_url}/sessions/implicit-sess/v1",
            api_key="dummy",
        )
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "test"}],
        )

        client = GatewayClient(gw_url)
        traces = client.get_session_traces("implicit-sess")
        assert len(traces) == 1
        assert traces[0].session_id == "implicit-sess"
        client.close()
