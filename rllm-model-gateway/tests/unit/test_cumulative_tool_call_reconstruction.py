"""Structured tool_call / reasoning reconstruction on the cumulative-token path.

In ``cumulative_token_mode`` the gateway bridges chat/completions to
``/v1/completions`` with a pre-tokenized prompt, so the serving stack's own chat
tool-call parser never runs. Text-protocol harnesses (Terminus-2) parse the raw
output themselves and send no OpenAI ``tools``. OpenAI function-calling clients
(opencode) send ``tools`` and need structured ``tool_calls`` back — without them
the agent sees only raw ``<think>…<tool_call>…`` text and never acts.

These tests pin: (1) a ``tools`` request gets structured content/reasoning/tool_calls
reconstructed via the renderer; (2) a request WITHOUT tools is byte-identical to the
old raw-text behavior; (3) missing/failing renderer falls back to raw text.

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
    """parse_response returns a reasoning block + one tool call, no plain content —
    the shape a Qwen3.5 reasoning turn that emits a tool call decodes to."""

    def parse_response(self, token_ids):
        return SimpleNamespace(
            content="",
            reasoning_content="let me think",
            tool_calls=[{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}],
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


def _completion_handler():
    async def handler(body):
        return {
            "id": "cmpl-x",
            "object": "text_completion",
            "choices": [
                {
                    "index": 0,
                    "text": "<think>let me think</think><tool_call>{...}</tool_call>",
                    "token_ids": [91, 92],
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.1, -0.2]},
                }
            ],
            "prompt_token_ids": body["prompt"],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 2},
        }

    return handler


_MSGS = [{"role": "user", "content": "x"}]
_TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a command"}}]


def _run_non_streaming(acc, request_body, bridged):
    proxy = _make_proxy(_completion_handler())
    resp = asyncio.run(
        proxy._handle_cumulative_non_streaming(
            _Request(),
            request_body,
            {"prompt": bridged, "add_special_tokens": False, "model": "q"},
            "sess1",
            acc,
            bridged,
        )
    )
    return json.loads(resp.body)


# --- unit: tool_call shape conversion ---------------------------------------


def test_to_openai_tool_calls_shapes_arguments_and_ids():
    out = _to_openai_tool_calls([{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}])
    assert out == [{"id": "call_0", "type": "function", "index": 0, "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]
    # Already-string arguments pass through unchanged.
    out2 = _to_openai_tool_calls([{"id": "abc", "function": {"name": "x", "arguments": "{}"}}])
    assert out2[0]["id"] == "abc" and out2[0]["function"]["arguments"] == "{}"


def test_assistant_message_raw_when_no_tools_or_no_renderer():
    # No tools requested -> raw content regardless of renderer.
    assert _assistant_message_from_completion("hi", [1, 2], {}, _FakeRenderer()) == {"role": "assistant", "content": "hi"}
    # Tools requested but no renderer -> raw fallback.
    assert _assistant_message_from_completion("hi", [1, 2], {"tools": _TOOLS}, None) == {"role": "assistant", "content": "hi"}


# --- non-streaming path ------------------------------------------------------


def test_cumulative_tools_reconstructs_structured_message():
    acc = TokenAccumulator(renderer=_FakeRenderer())
    acc.ingest_turn([1, 2, 3], [4, 5])
    body = _run_non_streaming(acc, {"messages": _MSGS, "tools": _TOOLS}, [1, 2, 3, 4, 5, 6, 7])

    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["reasoning_content"] == "let me think"
    assert msg["tool_calls"][0]["function"] == {"name": "bash", "arguments": '{"cmd": "ls"}'}
    assert msg["tool_calls"][0]["type"] == "function"
    assert "text" not in body["choices"][0]  # raw text popped, not leaked


def test_cumulative_no_tools_keeps_raw_content():
    """Terminus-2 style: no tools field -> byte-identical to prior raw behavior."""
    acc = TokenAccumulator(renderer=_FakeRenderer())  # renderer present but must be unused
    acc.ingest_turn([1, 2, 3], [4, 5])
    body = _run_non_streaming(acc, {"messages": _MSGS}, [1, 2, 3, 4, 5, 6, 7])

    msg = body["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "<think>let me think</think><tool_call>{...}</tool_call>"}
    assert "tool_calls" not in msg and "reasoning_content" not in msg


def test_cumulative_tools_no_renderer_falls_back_to_raw():
    acc = TokenAccumulator(renderer=None)
    acc.ingest_turn([1, 2, 3], [4, 5])
    body = _run_non_streaming(acc, {"messages": _MSGS, "tools": _TOOLS}, [1, 2, 3, 4, 5, 6, 7])
    assert body["choices"][0]["message"]["content"] == "<think>let me think</think><tool_call>{...}</tool_call>"
    assert "tool_calls" not in body["choices"][0]["message"]


# --- local streaming path ----------------------------------------------------


def test_cumulative_local_streaming_tools_emits_tool_calls():
    proxy = _make_proxy(_completion_handler())
    acc = TokenAccumulator(renderer=_FakeRenderer())
    acc.ingest_turn([1, 2, 3], [4, 5])
    bridged = [1, 2, 3, 4, 5, 6, 7]
    resp = asyncio.run(
        proxy._handle_cumulative_streaming_local(
            _Request(),
            {"messages": _MSGS, "tools": _TOOLS},
            {"prompt": bridged},
            "sess1",
            acc,
            bridged,
        )
    )
    chunks = []

    async def drain():
        async for c in resp.body_iterator:
            chunks.append(c if isinstance(c, str) else c.decode())

    asyncio.run(drain())
    joined = "".join(chunks)
    assert '"tool_calls"' in joined and '"bash"' in joined
    assert '"reasoning_content"' in joined
    assert chunks[-1].strip().endswith("[DONE]")
    # Training token capture is unaffected: the completion tokens still ingest.
    assert acc.turn_count == 2 and acc.prev_completion_ids == [91, 92]
