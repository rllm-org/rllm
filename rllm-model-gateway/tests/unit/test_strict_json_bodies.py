"""Lone surrogate escapes must survive the gateway intact.

JSON.stringify emits ``\\ud83d`` as pure ASCII whenever a string is sliced
through an astral character, e.g. truncated tool output. orjson rejects it,
but the stdlib and inference workers preserve it.
"""

import json

import httpx
import pytest
from rllm_model_gateway import GatewayConfig, create_app, fastjson

from tests.helpers.gateway_server import GatewayServer

SURROGATE_BODY = '{"model":"mock-model","messages":[{"role":"user","content":"tool out \\ud83d"}]}'


def test_surrogate_round_trip_matches_the_stdlib():
    body = '{"c":"x \\ud83d"}'
    assert json.loads(fastjson.dumps(fastjson.loads(body))) == json.loads(body)


@pytest.mark.parametrize("body", ['{"c":NaN}', '{"c":Infinity}', '{"c":-Infinity}', '{"c":1e400}'])
def test_load_fallback_does_not_accept_nonfinite_numbers(body):
    with pytest.raises(ValueError):
        fastjson.loads(body)


def test_malformed_bodies_still_raise():
    with pytest.raises(ValueError):
        fastjson.loads("not json")


def test_dumps_round_trips_a_lone_surrogate():
    """orjson refuses to emit one and UTF-8 cannot encode it, so the stdlib
    fallback must escape rather than raise (dumps_sorted keys message identity)."""
    payload = {"content": "x \ud83d"}
    for dump in (fastjson.dumps, fastjson.dumps_sorted):
        assert json.loads(dump(payload)) == payload


@pytest.fixture
def gateway(mock_vllm):
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,
        sync_traces=True,
    )
    server = GatewayServer(create_app(config), port=0)
    server.start()
    yield server
    server.stop()


def test_surrogate_body_is_traced_and_readable(gateway, mock_vllm):
    """End to end: the trace keeps its conversation, and reading it back works
    (Starlette's default renderer 500s on the stored surrogate)."""
    posted = httpx.post(
        f"{gateway.url}/sessions/s/v1/chat/completions",
        content=SURROGATE_BODY.encode(),
        headers={"content-type": "application/json"},
        timeout=30.0,
    )
    assert posted.status_code == 200
    expected = [{"role": "user", "content": "tool out \ud83d"}]
    assert mock_vllm.request_log[-1]["messages"] == expected

    traces = httpx.get(f"{gateway.url}/sessions/s/traces", timeout=30.0)
    assert traces.status_code == 200
    assert b"\\ud83d" in traces.content
    assert traces.json()[0]["messages"] == expected
