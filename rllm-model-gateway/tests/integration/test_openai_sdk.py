"""Integration: official OpenAI Python SDK against the gateway."""

from __future__ import annotations

import pytest
from rllm_model_gateway import GatewayClient

openai = pytest.importorskip("openai")


def test_openai_chat_completions_through_gateway(gateway_server):
    base_url, admin_key, agent_key = gateway_server
    client = GatewayClient(base_url, api_key=admin_key)
    sid = client.create_session("openai-chat")

    oa = openai.OpenAI(base_url=client.get_session_url(sid), api_key=agent_key)
    completion = oa.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    msg = completion.choices[0].message
    assert msg.content == "Hello from adapter"
    assert msg.tool_calls is not None and msg.tool_calls[0].function.name == "lookup"


def test_openai_chat_completions_streaming(gateway_server):
    base_url, admin_key, agent_key = gateway_server
    client = GatewayClient(base_url, api_key=admin_key)
    sid = client.create_session("openai-chat-stream")

    oa = openai.OpenAI(base_url=client.get_session_url(sid), api_key=agent_key)
    stream = oa.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    assert text == "Hello from adapter"


def test_openai_responses_through_gateway(gateway_server):
    base_url, admin_key, agent_key = gateway_server
    client = GatewayClient(base_url, api_key=admin_key)
    sid = client.create_session("openai-responses")

    oa = openai.OpenAI(base_url=client.get_session_url(sid), api_key=agent_key)
    response = oa.responses.create(model="m", input="hello")
    output_types = [item.type for item in response.output]
    assert output_types == ["reasoning", "message", "function_call"]
    text_items = [c.text for item in response.output if item.type == "message" for c in item.content if c.type == "output_text"]
    assert text_items == ["Hello from adapter"]


def test_openai_responses_streaming(gateway_server):
    base_url, admin_key, agent_key = gateway_server
    client = GatewayClient(base_url, api_key=admin_key)
    sid = client.create_session("openai-responses-stream")

    oa = openai.OpenAI(base_url=client.get_session_url(sid), api_key=agent_key)
    stream = oa.responses.create(model="m", input="hello", stream=True)
    accumulated_text = ""
    saw_completed = False
    for event in stream:
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            accumulated_text += event.delta
        if etype == "response.completed":
            saw_completed = True
    assert accumulated_text == "Hello from adapter"
    assert saw_completed


def test_traces_persisted_with_extras(gateway_server):
    base_url, admin_key, agent_key = gateway_server
    client = GatewayClient(base_url, api_key=admin_key)
    sid = client.create_session("traces")

    oa = openai.OpenAI(base_url=client.get_session_url(sid), api_key=agent_key)
    oa.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

    client.flush()  # drain async writes
    traces = client.get_session_traces(sid)
    assert len(traces) == 1
    extras = client.get_trace_extras(traces[0].trace_id)
    assert extras is not None
    fmt, data = extras
    import msgpack

    decoded = msgpack.unpackb(data, raw=False)
    assert decoded["completion_ids"] == [10, 20, 30, 40, 50, 60, 70]
