"""Structured tool_call / reasoning on the cumulative-token chat translation.

In ``cumulative_token_mode`` the gateway bridges chat/completions to ``/v1/completions``
with a pre-tokenized prompt, so the serving stack's own chat tool-call parser never runs.
OpenAI function-calling clients (opencode) send ``tools`` and need structured ``tool_calls``
back; text-protocol harnesses (Terminus-2) send none and parse raw output themselves.

Two producers feed tool_calls into the cumulative-turn translation, and both are covered:
  * the in-process handler (Fireworks/Tinker) parses and puts ``tool_calls``/``reasoning``
    on the choice — the gateway passes them through;
  * a raw vLLM worker returns only ``text`` — the gateway parses the completion tokens via
    the renderer when the client sent ``tools``.

Driven directly (no HTTP server) via ``asyncio.run``, mirroring
``test_cumulative_token_mode_local.py``.
"""

import asyncio
import json
from types import SimpleNamespace

from rllm_model_gateway.proxy import (
    ReverseProxy,
    _assistant_message_from_completion,
    _to_openai_tool_calls,
)
from rllm_model_gateway.store.memory_store import MemoryTraceStore
from rllm_model_gateway.token_accumulator import TokenAccumulator


class _State:
    weight_version = 0


class _Request:
    state = _State()


class _FakeRenderer:
    """parse_response returns reasoning + one tool call (prime-rl nested shape) — the
    shape a raw vLLM completion is parsed into on the HTTP-worker fallback path."""

    def parse_response(self, token_ids):
        return SimpleNamespace(
            content="",
            reasoning_content="let me think",
            tool_calls=[{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
        )


def _make_proxy(local_handler):
    return ReverseProxy(
        router=None,
        store=MemoryTraceStore(),
        sync_traces=True,
        local_handler=local_handler,
        cumulative_token_mode=True,
        renderer=None,
    )


# A handler that already parsed tool_calls onto the choice (the fixed tinker_adapter).
def _handler_with_tool_calls():
    async def handler(body):
        return {
            "id": "cmpl-x",
            "object": "text_completion",
            "choices": [
                {
                    "index": 0,
                    "text": "",  # parse-stripped; tool call carries the action
                    "token_ids": [91, 92],
                    "finish_reason": "tool_calls",
                    "reasoning": "let me think",
                    "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
                    "logprobs": {"token_logprobs": [-0.1, -0.2]},
                }
            ],
            "prompt_token_ids": body["prompt"],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 2},
        }

    return handler


# A raw-completion handler (text only, no tool_calls) — the vLLM-worker shape.
def _handler_text_only():
    async def handler(body):
        return {
            "id": "cmpl-x",
            "object": "text_completion",
            "choices": [{"index": 0, "text": "<tool_call>...</tool_call>", "token_ids": [91, 92], "finish_reason": "stop", "logprobs": {"token_logprobs": [-0.1, -0.2]}}],
            "prompt_token_ids": body["prompt"],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 2},
        }

    return handler


_MSGS = [{"role": "user", "content": "x"}]
_TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a command"}}]


def _run_non_streaming(handler, acc, request_body):
    proxy = _make_proxy(handler)
    resp = asyncio.run(
        proxy._handle_cumulative_non_streaming(
            _Request(),
            request_body,
            {"prompt": [1, 2, 3, 4, 5, 6, 7], "add_special_tokens": False, "model": "q"},
            "sess1",
            acc,
            [1, 2, 3, 4, 5, 6, 7],
        )
    )
    return json.loads(resp.body)


# --- unit: proxy _to_openai_tool_calls (nested shape) -----------------------


def test_to_openai_tool_calls_shapes_arguments_and_ids():
    out = _to_openai_tool_calls([{"function": {"name": "bash", "arguments": {"command": "ls"}}}])
    assert out == [{"id": "call_0", "type": "function", "index": 0, "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]


# --- unit: _assistant_message_from_completion (choice-based) -----------------


def test_message_prefers_handler_provided_tool_calls():
    tc = [{"id": "call_0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]
    choice = {"text": "", "tool_calls": tc, "reasoning": "thinking"}
    msg = _assistant_message_from_completion(choice, [1, 2], {"tools": _TOOLS}, _FakeRenderer())
    assert msg["tool_calls"] == tc  # passed through verbatim, not re-parsed
    assert msg["reasoning"] == "thinking"


def test_message_parses_tokens_when_handler_gave_only_text():
    choice = {"text": "<tool_call>...</tool_call>"}
    msg = _assistant_message_from_completion(choice, [1, 2], {"tools": _TOOLS}, _FakeRenderer())
    assert msg["tool_calls"][0]["function"] == {"name": "bash", "arguments": '{"command": "ls"}'}
    assert msg["reasoning_content"] == "let me think"


def test_message_raw_when_no_tools_or_no_renderer():
    assert _assistant_message_from_completion({"text": "hi"}, [1, 2], {}, _FakeRenderer()) == {"role": "assistant", "content": "hi"}
    assert _assistant_message_from_completion({"text": "hi"}, [1, 2], {"tools": _TOOLS}, None) == {"role": "assistant", "content": "hi"}


# --- non-streaming path ------------------------------------------------------


def test_cumulative_passes_through_handler_tool_calls():
    acc = TokenAccumulator(renderer=None)  # renderer unused: handler already parsed
    acc.ingest_turn([1, 2, 3], [4, 5])
    body = _run_non_streaming(_handler_with_tool_calls(), acc, {"messages": _MSGS, "tools": _TOOLS})
    msg = body["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"] == {"name": "bash", "arguments": '{"command": "ls"}'}
    assert msg["reasoning"] == "let me think"
    assert "text" not in body["choices"][0] and "tool_calls" not in body["choices"][0]


def test_cumulative_parses_raw_completion_when_tools_requested():
    acc = TokenAccumulator(renderer=_FakeRenderer())
    acc.ingest_turn([1, 2, 3], [4, 5])
    body = _run_non_streaming(_handler_text_only(), acc, {"messages": _MSGS, "tools": _TOOLS})
    msg = body["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"] == {"name": "bash", "arguments": '{"command": "ls"}'}


def test_cumulative_no_tools_keeps_raw_content():
    acc = TokenAccumulator(renderer=_FakeRenderer())  # present but must be unused
    acc.ingest_turn([1, 2, 3], [4, 5])
    body = _run_non_streaming(_handler_text_only(), acc, {"messages": _MSGS})
    msg = body["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "<tool_call>...</tool_call>"}
    assert "tool_calls" not in msg


# --- local streaming path ----------------------------------------------------


def test_cumulative_local_streaming_emits_tool_calls():
    proxy = _make_proxy(_handler_with_tool_calls())
    acc = TokenAccumulator(renderer=None)
    acc.ingest_turn([1, 2, 3], [4, 5])
    bridged = [1, 2, 3, 4, 5, 6, 7]
    resp = asyncio.run(proxy._handle_cumulative_streaming_local(_Request(), {"messages": _MSGS, "tools": _TOOLS}, {"prompt": bridged}, "sess1", acc, bridged))
    chunks = []

    async def drain():
        async for c in resp.body_iterator:
            chunks.append(c if isinstance(c, str) else c.decode())

    asyncio.run(drain())
    joined = "".join(chunks)
    assert '"tool_calls"' in joined and '"bash"' in joined
    assert chunks[-1].strip().endswith("[DONE]")
    # Training token capture unaffected: the completion tokens still ingest.
    assert acc.turn_count == 2 and acc.prev_completion_ids == [91, 92]
