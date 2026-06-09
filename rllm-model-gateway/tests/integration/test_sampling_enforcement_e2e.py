"""E2E proof of the gateway "one rule": session sampling params always win.

Runs a real gateway (thread) in front of a mock vLLM that records every
forwarded request body. No GPU, no pytest_asyncio — uses the sync OpenAI client.
Asserts that, for a session created with sampling params:
  * a configured key overrides the client's conflicting value (gateway wins);
  * a configured extra key (presence_penalty) is injected;
  * an unconfigured key the client sent passes through untouched.
And that with no session params, client values pass through unchanged.
"""

from __future__ import annotations

import openai
import pytest
from rllm_model_gateway import GatewayClient, GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer
from tests.helpers.mock_vllm import MockVLLMServer


@pytest.fixture
def mock_upstream():
    server = MockVLLMServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def gateway(mock_upstream):
    # External-provider-style gateway: no vLLM-only field injection.
    cfg = GatewayConfig(
        workers=[{"url": mock_upstream.url}],
        add_logprobs=False,
        add_return_token_ids=False,
    )
    server = GatewayServer(create_app(cfg))
    server.start()
    yield server, mock_upstream
    server.stop()


def test_session_params_override_client_and_inject_extra(gateway):
    server, mock_upstream = gateway
    client = GatewayClient(server.url)
    sid = client.create_session(
        session_id="enforce",
        sampling_params={"temperature": 0.123, "presence_penalty": 0.7},
    )
    oai = openai.OpenAI(base_url=client.get_session_url(sid), api_key="dummy")

    oai.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.9,  # conflicts with session config → gateway must win
        top_p=0.5,  # not configured → must pass through
    )

    fwd = mock_upstream.request_log[-1]
    assert fwd["temperature"] == 0.123, "gateway session config must override client temperature"
    assert fwd["presence_penalty"] == 0.7, "configured extra key must be injected"
    assert fwd["top_p"] == 0.5, "unconfigured key must pass through untouched"
    client.close()


def test_no_session_params_passes_client_values_through(gateway):
    server, mock_upstream = gateway
    client = GatewayClient(server.url)
    sid = client.create_session(session_id="passthrough")  # no sampling params
    oai = openai.OpenAI(base_url=client.get_session_url(sid), api_key="dummy")

    oai.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.9,
        top_p=0.5,
    )

    fwd = mock_upstream.request_log[-1]
    assert fwd["temperature"] == 0.9
    assert fwd["top_p"] == 0.5
    assert "presence_penalty" not in fwd
    client.close()
