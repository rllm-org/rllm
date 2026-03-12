"""End-to-end training workflow tests against real vLLM.

Requires real vLLM on localhost:4000 (auto-skipped if unavailable).
"""

import httpx
import litellm
import openai
import pytest
from rllm_model_gateway import GatewayClient

from .conftest import VLLM_BASE, requires_vllm


def _get_model_id() -> str:
    try:
        r = httpx.get(f"{VLLM_BASE}/v1/models", timeout=3.0)
        return r.json()["data"][0]["id"]
    except Exception:
        return "unknown"


@pytest.fixture(scope="module")
def model_id():
    return _get_model_id()


@requires_vllm
class TestSingleRolloutLifecycle:
    def test_single_rollout_lifecycle(self, gateway_server_url: str, model_id: str):
        """Full flow with real vLLM model responses and real token IDs."""

        client = GatewayClient(gateway_server_url)
        sid = client.create_session(session_id="e2e-rollout")
        session_url = client.get_session_url(sid)
        oai = openai.OpenAI(base_url=session_url, api_key="dummy")

        resp = oai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=20,
        )
        assert len(resp.choices[0].message.content) > 0

        traces = client.get_session_traces(sid)
        assert len(traces) == 1
        trace = traces[0]
        assert trace.session_id == sid
        assert trace.model == model_id
        assert len(trace.prompt_token_ids) > 0
        assert len(trace.completion_token_ids) > 0
        assert trace.logprobs is not None
        assert len(trace.logprobs) > 0
        assert trace.response_message["role"] == "assistant"
        assert len(trace.response_message["content"]) > 0
        assert trace.finish_reason in ("stop", "length")
        assert trace.latency_ms > 0
        client.close()


@requires_vllm
class TestMultiTurnAgentConversation:
    def test_multi_turn_agent_conversation(self, gateway_server_url: str, model_id: str):
        """Multi-turn with real model, verify traces have real logprobs."""

        client = GatewayClient(gateway_server_url)
        sid = client.create_session(session_id="e2e-multi-turn")
        session_url = client.get_session_url(sid)
        oai = openai.OpenAI(base_url=session_url, api_key="dummy")

        for i in range(3):
            oai.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": f"Count: {i}"}],
                max_tokens=10,
            )

        traces = client.get_session_traces(sid)
        assert len(traces) == 3
        for trace in traces:
            assert trace.session_id == sid
            assert len(trace.prompt_token_ids) > 0
            assert len(trace.completion_token_ids) > 0
            assert trace.logprobs is not None
            assert len(trace.logprobs) > 0
        client.close()


@requires_vllm
class TestStreamingRollout:
    def test_streaming_rollout(self, gateway_server_url: str, model_id: str):
        """Streaming with real model, verify accumulated tokens match."""

        client = GatewayClient(gateway_server_url)
        sid = client.create_session(session_id="e2e-stream")
        session_url = client.get_session_url(sid)
        oai = openai.OpenAI(base_url=session_url, api_key="dummy")

        stream = oai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=10,
            stream=True,
        )
        content_parts = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
        full_content = "".join(content_parts)
        assert len(full_content) > 0

        traces = client.get_session_traces(sid)
        assert len(traces) == 1
        trace = traces[0]
        assert len(trace.prompt_token_ids) > 0
        assert len(trace.completion_token_ids) > 0
        assert trace.logprobs is not None
        assert len(trace.logprobs) > 0
        assert trace.response_message["role"] == "assistant"
        client.close()


@requires_vllm
class TestLiteLLMClientE2E:
    def test_litellm_client_e2e(self, gateway_server_url: str, model_id: str):
        """LiteLLM completion() through gateway to real vLLM, verify traces captured."""
        client = GatewayClient(gateway_server_url)
        sid = client.create_session(session_id="e2e-litellm")
        session_url = client.get_session_url(sid)

        resp = litellm.completion(
            model=f"openai/{model_id}",
            messages=[{"role": "user", "content": "Say hello."}],
            api_base=session_url,
            api_key="dummy",
            max_tokens=20,
        )
        assert len(resp.choices[0].message.content) > 0

        traces = client.get_session_traces(sid)
        assert len(traces) == 1
        assert traces[0].session_id == sid
        assert len(traces[0].prompt_token_ids) > 0
        client.close()
