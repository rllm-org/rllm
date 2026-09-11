"""Whitespace heartbeat for slow non-streaming completions.

Middleboxes on the response path (Cloudflare quick tunnel: 120s read timer;
ngrok: ~300s / ERR_NGROK_3004; NAT flow tables) silently kill responses that
stay byte-silent while the model generates. ``_handle_non_streaming`` commits
a chunked 200 after ``heartbeat_initial_delay_s`` and emits a single space —
insignificant JSON leading whitespace — every ``heartbeat_interval_s`` until
the upstream body is ready.

Invariants pinned here:
* the parsed response is byte-for-byte identical to the upstream payload
  (spaces are stripped by every JSON parser);
* fast responses — successes AND errors — bypass the heartbeat entirely and
  keep their true status codes;
* an upstream failure after the 200 is committed still surfaces as a
  parseable OpenAI-style error object (never a silent hang);
* a genuinely hung upstream is bounded by ``heartbeat_budget_s``.
"""

from __future__ import annotations

import asyncio
import json

from rllm_model_gateway.proxy import ReverseProxy
from rllm_model_gateway.store.memory_store import MemoryTraceStore


class _State:
    weight_version = 0
    session_id = "sess-hb"
    originally_requested_logprobs = True


class _URL:
    path = "/v1/chat/completions"
    query = ""


class _Request:
    state = _State()
    url = _URL()
    method = "POST"
    headers: dict[str, str] = {"content-type": "application/json"}

    async def body(self) -> bytes:
        return json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()


_COMPLETION = {
    "id": "chatcmpl-hb",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def _make_proxy(local_handler, *, delay=0.2, interval=0.1, budget=60.0):
    return ReverseProxy(
        router=None,
        store=MemoryTraceStore(),
        sync_traces=True,
        local_handler=local_handler,
        heartbeat_initial_delay_s=delay,
        heartbeat_interval_s=interval,
        heartbeat_budget_s=budget,
    )


async def _drain(response) -> bytes:
    """Collect a (Streaming)Response's full body bytes."""
    if hasattr(response, "body_iterator"):
        chunks = [chunk async for chunk in response.body_iterator]
        return b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)
    return bytes(response.body)


async def _handle_and_drain(proxy):
    response = await proxy.handle(_Request())
    return response, await _drain(response)


def test_slow_response_heartbeats_and_parses_identically():
    """The regression case: generation slower than every middlebox timer.

    Bytes must flow before the body is ready, and the parsed payload must be
    exactly what the upstream produced.
    """

    async def slow_handler(body):
        await asyncio.sleep(1.0)  # >> delay + several intervals
        return dict(_COMPLETION)

    proxy = _make_proxy(slow_handler)
    response, raw = asyncio.run(_handle_and_drain(proxy))

    assert response.status_code == 200
    assert raw.startswith(b" "), "no heartbeat bytes were emitted before the body"
    assert raw.lstrip(b" ").startswith(b"{"), "payload must follow the whitespace"
    assert json.loads(raw) == _COMPLETION, "parsed response must be identical to the upstream payload"


def test_fast_response_is_untouched():
    """Fast generations never enter heartbeat mode: plain body, no padding."""

    async def fast_handler(body):
        return dict(_COMPLETION)

    proxy = _make_proxy(fast_handler, delay=5.0)
    response, raw = asyncio.run(_handle_and_drain(proxy))

    assert response.status_code == 200
    assert not raw.startswith(b" ")
    assert json.loads(raw) == _COMPLETION


def test_fast_upstream_error_keeps_true_status():
    """An upstream that fails quickly keeps its real status code — the
    heartbeat must not blur errors it never needed to paper over."""

    class _Resp:
        status_code = 429
        content = json.dumps({"error": {"message": "rate limited"}}).encode()

    proxy = _make_proxy(None, delay=5.0)

    async def fake_send(**kwargs):
        return _Resp()

    proxy._send_with_retry = fake_send
    proxy.router = type("R", (), {"route": lambda self, sid: type("W", (), {"api_url": "http://up", "url": "http://up"})(), "release": lambda self, url: None})()

    response, raw = asyncio.run(_handle_and_drain(proxy))
    assert response.status_code == 429
    assert json.loads(raw)["error"]["message"] == "rate limited"


def test_upstream_failure_after_commit_surfaces_as_error_object():
    """Failure vs hang, case 1: the upstream *fails* mid-heartbeat. The client
    gets a parseable OpenAI-style error object immediately — not a hang, not
    a truncated stream."""

    async def dying_handler(body):
        await asyncio.sleep(0.6)
        raise RuntimeError("upstream exploded")

    proxy = _make_proxy(dying_handler)
    _, raw = asyncio.run(_handle_and_drain(proxy))

    payload = json.loads(raw)
    assert payload["error"]["type"] == "gateway_upstream_error"
    assert "upstream exploded" in payload["error"]["message"]


def test_hung_upstream_is_bounded_by_budget():
    """Failure vs hang, case 2: the upstream never answers. The heartbeat does
    not keep the client waiting forever — the budget converts the hang into a
    typed timeout error object."""

    async def hung_handler(body):
        await asyncio.sleep(3600)

    proxy = _make_proxy(hung_handler, budget=0.8)
    _, raw = asyncio.run(_handle_and_drain(proxy))

    payload = json.loads(raw)
    assert payload["error"]["type"] == "gateway_upstream_timeout"
    assert payload["error"]["code"] == 504


def test_heartbeat_disabled_restores_plain_behavior():
    async def slow_handler(body):
        await asyncio.sleep(0.3)
        return dict(_COMPLETION)

    proxy = _make_proxy(slow_handler, interval=0)  # disabled
    _, raw = asyncio.run(_handle_and_drain(proxy))
    assert not raw.startswith(b" ")
    assert json.loads(raw) == _COMPLETION
