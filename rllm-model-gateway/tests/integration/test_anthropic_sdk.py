"""Integration: official Anthropic Python SDK against the gateway."""

from __future__ import annotations

import pytest
from rllm_model_gateway import GatewayClient

anthropic = pytest.importorskip("anthropic")


def test_anthropic_messages_through_gateway(gateway_server):
    base_url, admin_key, agent_key = gateway_server
    client = GatewayClient(base_url, api_key=admin_key)
    sid = client.create_session("anth")

    cl = anthropic.Anthropic(base_url=client.get_anthropic_session_url(sid), api_key=agent_key)
    msg = cl.messages.create(
        model="claude-fake",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    block_types = [b.type for b in msg.content]
    assert "text" in block_types
    assert "tool_use" in block_types
    text_blocks = [b for b in msg.content if b.type == "text"]
    assert text_blocks[0].text == "Hello from adapter"
    tu_blocks = [b for b in msg.content if b.type == "tool_use"]
    assert tu_blocks[0].name == "lookup"
    assert tu_blocks[0].input == {"q": "rllm"}
    assert msg.stop_reason == "tool_use"


def test_anthropic_messages_streaming(gateway_server):
    base_url, admin_key, agent_key = gateway_server
    client = GatewayClient(base_url, api_key=admin_key)
    sid = client.create_session("anth-stream")

    cl = anthropic.Anthropic(base_url=client.get_anthropic_session_url(sid), api_key=agent_key)
    accumulated_text = ""
    with cl.messages.stream(
        model="claude-fake",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    ) as stream:
        for text in stream.text_stream:
            accumulated_text += text
    assert accumulated_text == "Hello from adapter"
