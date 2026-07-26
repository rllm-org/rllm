"""Per-turn ``max_tokens`` is clamped to the prompt's remaining context headroom.

vLLM's OpenAI server rejects ``prompt_tokens + max_tokens > max_model_len`` with
a 400 instead of truncating, so a cumulative prompt that still fits the window
fails anyway once the per-turn cap pushes the total over it. VERL's own
token-in-token-out path clamps before sampling (``vllm_async_server.generate``),
and the gateway is the equivalent chokepoint on the HTTP path.
"""

import asyncio
import json
from types import SimpleNamespace

from rllm_model_gateway.proxy import ReverseProxy
from rllm_model_gateway.store.memory_store import MemoryTraceStore


class _Request:
    method = "POST"

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()
        self.state = SimpleNamespace(session_id="sess1", originally_requested_logprobs=False, weight_version=0)
        self.url = SimpleNamespace(path="/v1/chat/completions", query="")

    async def body(self) -> bytes:
        return self._payload


def _proxy(*, max_model_len=None):
    return ReverseProxy(
        router=None,
        store=MemoryTraceStore(),
        sync_traces=True,
        cumulative_token_mode=True,
        max_model_len=max_model_len,
    )


def _sent_body(*, prompt_len, max_tokens, max_model_len):
    """Run one cumulative turn and return the body handed to the worker."""
    proxy = _proxy(max_model_len=max_model_len)
    captured: dict = {}

    async def _non_streaming(request, request_body, completions_body, session_id, acc, token_ids, originally_requested_logprobs=False, *, replay=False):
        captured.update(completions_body)
        return "sent"

    proxy._handle_cumulative_non_streaming = _non_streaming
    body = {"model": "q", "messages": [{"role": "user", "content": "x"}], "max_tokens": max_tokens}
    asyncio.run(proxy._handle_cumulative_turn(_Request(body), body, "sess1", None, list(range(prompt_len))))
    return captured


def test_long_prompt_shrinks_max_tokens_to_the_remaining_window():
    body = _sent_body(prompt_len=55710, max_tokens=16384, max_model_len=67584)

    assert body["max_tokens"] == 67584 - 55710


def test_short_prompt_keeps_the_requested_cap():
    body = _sent_body(prompt_len=4096, max_tokens=16384, max_model_len=67584)

    assert body["max_tokens"] == 16384


def test_prompt_at_the_window_still_asks_for_one_token():
    # vLLM >=0.20 raises on max_tokens < 1, so VERL floors at 1 and lets the
    # server reject the over-long prompt itself. Same behaviour here.
    body = _sent_body(prompt_len=67584, max_tokens=16384, max_model_len=67584)

    assert body["max_tokens"] == 1


def test_no_clamping_without_a_configured_window():
    body = _sent_body(prompt_len=55710, max_tokens=16384, max_model_len=None)

    assert body["max_tokens"] == 16384


def test_absent_max_tokens_is_left_for_the_server_to_derive():
    proxy = _proxy(max_model_len=67584)
    body: dict = {"prompt": [1, 2, 3]}

    proxy._clamp_max_tokens(body, 55710)

    assert "max_tokens" not in body
