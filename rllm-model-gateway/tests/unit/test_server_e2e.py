"""End-to-end server tests via FastAPI TestClient."""

from __future__ import annotations

import msgpack
from rllm_model_gateway import (
    NormalizedRequest,
    NormalizedResponse,
    ToolCall,
    Usage,
)


async def _adapter(req: NormalizedRequest) -> NormalizedResponse:
    return NormalizedResponse(
        content="answer",
        reasoning="thought",
        tool_calls=[ToolCall(id="c1", name="t", arguments='{"x":1}')],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        extras={"completion_ids": [1, 2, 3, 4, 5], "logprobs": [-0.1] * 5},
    )


def test_session_required(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        # Direct hit on /v1 endpoint without /sessions prefix should 400.
        r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert r.status_code == 400


def test_chat_completions_round_trip(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg["content"] == "answer"
        assert msg["tool_calls"][0]["function"]["name"] == "t"


def test_responses_round_trip(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post("/sessions/s1/v1/responses", json={"model": "m", "input": "hi"})
        assert r.status_code == 200
        types = [item["type"] for item in r.json()["output"]]
        assert types == ["reasoning", "message", "function_call"]


def test_anthropic_round_trip(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/messages",
            json={
                "model": "m",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        types = [b["type"] for b in r.json()["content"]]
        assert types == ["thinking", "text", "tool_use"]


def test_traces_persist_with_extras(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        client.post("/admin/flush")  # drain async writes
        traces = client.get("/sessions/s1/traces").json()
        assert len(traces) == 1
        assert traces[0]["content"] == "answer"
        tid = traces[0]["trace_id"]
        r = client.get(f"/traces/{tid}/extras")
        assert r.status_code == 200
        extras = msgpack.unpackb(r.content, raw=False)
        assert extras["completion_ids"] == [1, 2, 3, 4, 5]


def test_streaming_emits_done(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        with client.stream(
            "POST",
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            data_lines = [line for line in r.iter_lines() if line.startswith("data: ")]
            assert data_lines[-1] == "data: [DONE]"


def test_streaming_responses_emits_completed(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        with client.stream(
            "POST",
            "/sessions/s1/v1/responses",
            json={
                "model": "m",
                "input": "hi",
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            event_names = [line[len("event: ") :] for line in r.iter_lines() if line.startswith("event: ")]
            assert event_names[0] == "response.created"
            assert event_names[-1] == "response.completed"


def test_adapter_exception_returns_502(gateway_app_factory):
    async def angry_adapter(req):
        raise RuntimeError("boom")

    with gateway_app_factory(adapter=angry_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 502
        assert "Adapter error" in r.json()["error"]


def test_invalid_json_body_handled(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        # Malformed JSON body — gateway should still accept gracefully (treat as empty).
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        # Either 200 (treated as empty body) or 400 (rejected); not a 500.
        assert r.status_code < 500


def test_responses_previous_response_id_rejected(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/responses",
            json={
                "model": "m",
                "input": "hi",
                "previous_response_id": "resp_x",
            },
        )
        assert r.status_code == 400


def test_unknown_session_returns_404(gateway_app_factory):
    """Sessions must be created explicitly; hitting an unknown session 404s."""
    with gateway_app_factory(adapter=_adapter) as client:
        r = client.post(
            "/sessions/never-created/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 404
        assert "not found" in r.json()["error"].lower()


def test_get_trace_endpoint(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        client.post("/sessions/s1/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        client.post("/admin/flush")
        traces = client.get("/sessions/s1/traces").json()
        tid = traces[0]["trace_id"]

        r = client.get(f"/traces/{tid}")
        assert r.status_code == 200
        assert r.json()["trace_id"] == tid

        r = client.get("/traces/nonexistent")
        assert r.status_code == 404


def test_list_and_delete_session_endpoints(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        client.post("/sessions", json={"session_id": "s2"})

        r = client.get("/sessions")
        sids = {s["session_id"] for s in r.json()}
        assert {"s1", "s2"}.issubset(sids)

        client.post("/sessions/s1/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        client.post("/admin/flush")
        r = client.delete("/sessions/s1")
        assert r.status_code == 200
        assert r.json()["deleted"] >= 1


def test_admin_flush_endpoint(gateway_app_factory):
    with gateway_app_factory(adapter=_adapter) as client:
        r = client.post("/admin/flush")
        assert r.status_code == 200
        assert r.json()["status"] == "flushed"


def test_trace_extras_404_when_absent(gateway_app_factory):
    async def no_extras_adapter(req):
        return NormalizedResponse(content="hi", finish_reason="stop")

    with gateway_app_factory(adapter=no_extras_adapter) as client:
        client.post("/sessions", json={"session_id": "s1"})
        client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        client.post("/admin/flush")
        traces = client.get("/sessions/s1/traces").json()
        tid = traces[0]["trace_id"]
        assert client.get(f"/traces/{tid}/extras").status_code == 404


# ---------------------------------------------------------------------------
# Sampling-params priority + model pin
# ---------------------------------------------------------------------------


def test_model_pin_overrides_request_model(gateway_app_factory):
    captured = {}

    async def capture_adapter(req: NormalizedRequest) -> NormalizedResponse:
        captured["kwargs"] = dict(req.kwargs)
        return NormalizedResponse(content="ok", finish_reason="stop")

    with gateway_app_factory(adapter=capture_adapter, model="pinned-model") as client:
        client.post("/sessions", json={"session_id": "s1"})
        r = client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "client-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        # Pinned model is recorded in the trace
        client.post("/admin/flush")
        traces = client.get("/sessions/s1/traces").json()
        assert traces[0]["model"] == "pinned-model"


def test_session_sampling_priority_client_wins(gateway_app_factory):
    captured = {}

    async def capture(req: NormalizedRequest) -> NormalizedResponse:
        captured["kwargs"] = dict(req.kwargs)
        return NormalizedResponse(content="ok", finish_reason="stop")

    with gateway_app_factory(adapter=capture, sampling_params_priority="client") as client:
        client.post(
            "/sessions",
            json={
                "session_id": "s1",
                "sampling_params": {"temperature": 0.1, "max_tokens": 999},
            },
        )
        client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.9,  # client value should win
            },
        )
        # Client wins on overlap, session fills missing.
        assert captured["kwargs"]["temperature"] == 0.9
        assert captured["kwargs"]["max_tokens"] == 999


def test_session_sampling_priority_session_wins(gateway_app_factory):
    captured = {}

    async def capture(req: NormalizedRequest) -> NormalizedResponse:
        captured["kwargs"] = dict(req.kwargs)
        return NormalizedResponse(content="ok", finish_reason="stop")

    with gateway_app_factory(adapter=capture, sampling_params_priority="session") as client:
        client.post(
            "/sessions",
            json={
                "session_id": "s1",
                "sampling_params": {"temperature": 0.1, "max_tokens": 999},
            },
        )
        client.post(
            "/sessions/s1/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.9,
            },
        )
        # Session value wins on conflict.
        assert captured["kwargs"]["temperature"] == 0.1
        assert captured["kwargs"]["max_tokens"] == 999
