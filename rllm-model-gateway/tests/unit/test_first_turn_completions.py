"""A session's opening turn takes the same ``/v1/completions`` path as later turns.

Turn 0 has no prior tokens to bridge from, so it used to fall through to a raw
chat pass-through. Against an HTTP worker that leaves the renderer out of the
loop for that one call: it neither renders the prompt nor parses the response,
so a function-calling client (mini-swe-agent sends ``tools`` on every request)
gets its tool call as plain text and rejects the turn as a format error.
Rendering the opening messages from scratch keeps rendering AND extraction on
the renderer for every turn. Backends with an in-process handler parse their own
responses, so they keep the chat path.
"""

import asyncio
import json
from types import SimpleNamespace

from rllm_model_gateway.proxy import ReverseProxy
from rllm_model_gateway.store.memory_store import MemoryTraceStore
from rllm_model_gateway.token_accumulator import TokenAccumulator


class _FakeRenderer:
    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        assert add_generation_prompt, "an opening prompt must end with the generation prompt"
        return [1, 2, 3] + ([9] if tools else [])


class _Request:
    method = "POST"

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()
        self.state = SimpleNamespace(session_id="sess1", originally_requested_logprobs=False, weight_version=0)
        self.url = SimpleNamespace(path="/v1/chat/completions", query="")

    async def body(self) -> bytes:
        return self._payload


def _proxy(*, renderer, local_handler=None):
    proxy = ReverseProxy(
        router=None,
        store=MemoryTraceStore(),
        sync_traces=True,
        local_handler=local_handler,
        cumulative_token_mode=True,
        renderer=renderer,
    )

    routed: dict = {}

    async def _cumulative(request, request_body, session_id, acc, token_ids, originally_requested_logprobs=False, *, replay=False):
        routed["path"] = "completions"
        routed["token_ids"] = token_ids
        return "cumulative"

    async def _chat(request, raw_body, request_body, session_id, originally_requested_logprobs=False):
        routed["path"] = "chat"
        return "chat"

    async def _started():
        return None

    proxy._handle_cumulative_turn = _cumulative
    proxy._handle_non_streaming = _chat
    proxy._ensure_started = _started
    return proxy, routed


_TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a command"}}]
_BODY = {"model": "q", "messages": [{"role": "user", "content": "solve it"}], "tools": _TOOLS}


def test_first_turn_is_rendered_and_sent_to_completions():
    proxy, routed = _proxy(renderer=_FakeRenderer())

    assert asyncio.run(proxy.handle(_Request(_BODY))) == "cumulative"
    assert routed["path"] == "completions"
    assert routed["token_ids"] == [1, 2, 3, 9]  # tools folded into the rendered prompt


def test_first_turn_keeps_chat_path_for_in_process_handlers():
    async def handler(body):
        return {}

    proxy, routed = _proxy(renderer=_FakeRenderer(), local_handler=handler)

    assert asyncio.run(proxy.handle(_Request(_BODY))) == "chat"
    assert routed["path"] == "chat"


def test_first_turn_falls_back_to_chat_when_the_renderer_cannot_render():
    proxy, routed = _proxy(renderer=object())  # no render_ids

    assert asyncio.run(proxy.handle(_Request(_BODY))) == "chat"
    assert routed["path"] == "chat"


def test_build_initial_prompt_returns_none_without_a_renderer():
    assert TokenAccumulator(renderer=None).build_initial_prompt([{"role": "user", "content": "x"}]) is None
