"""Cumulative token mode over the in-process ``local_handler`` (e.g. Tinker).

The HTTP-worker path is covered by ``test_cumulative_token_mode.py``. These
tests cover the local-handler branch added so backends without a vLLM
``/v1/completions`` worker (Tinker runs in-process) get the same drift-free
prefix-extension: turn 2+ samples directly from pre-tokenized prompt IDs built
by the renderer, and the accumulator ingests the result.

Tests drive the proxy methods directly (no HTTP server) via ``asyncio.run`` to
avoid a pytest-asyncio dependency.
"""

import asyncio
import json

from rllm_model_gateway.proxy import ReverseProxy
from rllm_model_gateway.store.memory_store import MemoryTraceStore
from rllm_model_gateway.token_accumulator import TokenAccumulator


class _State:
    weight_version = 0


class _Request:
    """Minimal stand-in: the cumulative-local paths only read request.state."""

    state = _State()


class _ParsedResponse:
    content = "parsed action"
    reasoning_content = "parsed reasoning"
    tool_calls = [{"name": "bash", "arguments": {"command": "ls -la"}}]


class _ParsingRenderer:
    def __init__(self):
        self.seen_token_ids = None

    def parse_response(self, token_ids):
        self.seen_token_ids = list(token_ids)
        return _ParsedResponse()


def _make_proxy(local_handler, renderer=None):
    """ReverseProxy wired for the local cumulative path (no router/worker)."""
    return ReverseProxy(
        router=None,
        store=MemoryTraceStore(),
        sync_traces=True,
        local_handler=local_handler,
        cumulative_token_mode=True,
        renderer=renderer,  # accumulator state is set directly in these tests
    )


def _completion_handler(record):
    """Fake Tinker token-path handler: echoes ``prompt`` as prompt_token_ids,
    returns a fixed 2-token completion. Records the body it was called with."""

    async def handler(body):
        record.append(body)
        return {
            "id": "cmpl-x",
            "object": "text_completion",
            "choices": [
                {
                    "index": 0,
                    "text": "next action",
                    "token_ids": [91, 92],
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.1, -0.2]},
                }
            ],
            "prompt_token_ids": body["prompt"],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 2},
        }

    return handler


def test_cumulative_local_non_streaming_ingests_and_translates():
    record = []
    proxy = _make_proxy(_completion_handler(record))

    acc = TokenAccumulator(renderer=None)
    acc.ingest_turn([1, 2, 3], [4, 5])  # turn 1: prompt + completion already captured
    bridged = [1, 2, 3, 4, 5, 6, 7]  # what the renderer would produce for turn 2

    completions_body = {"prompt": bridged, "add_special_tokens": False, "model": "q"}
    resp = asyncio.run(
        proxy._handle_cumulative_non_streaming(
            _Request(),
            {"messages": [{"role": "user", "content": "x"}]},
            completions_body,
            "sess1",
            acc,
            bridged,
        )
    )

    # The local handler was called with the pre-tokenized prompt, not messages.
    assert record and record[0]["prompt"] == bridged
    assert "messages" not in record[0]

    # Response is translated back to chat format for the agent.
    body = json.loads(resp.body)
    assert resp.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "next action"}

    # Accumulator advanced: prefix-extension holds (bridged starts with prev prompt+completion).
    assert acc.turn_count == 2
    assert acc.prev_prompt_ids == bridged
    assert acc.prev_completion_ids == [91, 92]
    assert acc.cumulative_ids[: len([1, 2, 3, 4, 5])] == [1, 2, 3, 4, 5]


def test_cumulative_local_non_streaming_reconstructs_structured_chat_message():
    record = []
    renderer = _ParsingRenderer()
    proxy = _make_proxy(_completion_handler(record), renderer=renderer)

    acc = TokenAccumulator(renderer=None)
    acc.ingest_turn([1, 2, 3], [4, 5])
    bridged = [1, 2, 3, 4, 5, 6, 7]

    resp = asyncio.run(
        proxy._handle_cumulative_non_streaming(
            _Request(),
            {"messages": [{"role": "user", "content": "x"}]},
            {"prompt": bridged, "add_special_tokens": False, "model": "q"},
            "sess1",
            acc,
            bridged,
        )
    )

    body = json.loads(resp.body)
    message = body["choices"][0]["message"]
    assert renderer.seen_token_ids == [91, 92]
    assert message["content"] == "parsed action"
    assert message["reasoning"] == "parsed reasoning"
    assert message["tool_calls"] == [{"id": "call_0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls -la"}'}}]
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_cumulative_local_streaming_emits_sse_and_ingests():
    record = []
    proxy = _make_proxy(_completion_handler(record))

    acc = TokenAccumulator(renderer=None)
    acc.ingest_turn([1, 2, 3], [4, 5])
    bridged = [1, 2, 3, 4, 5, 6, 7]

    resp = asyncio.run(
        proxy._handle_cumulative_streaming_local(
            _Request(),
            {"messages": [{"role": "user", "content": "x"}]},
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

    assert any('"role": "assistant"' in c for c in chunks)
    assert any('"next action"' in c for c in chunks)
    assert chunks[-1].strip().endswith("[DONE]")
    # Same ingest as non-streaming.
    assert acc.turn_count == 2 and acc.prev_completion_ids == [91, 92]


def test_cumulative_local_replay_regenerates_in_place_without_advancing():
    """A duplicate resend regenerates (fresh sample) and overwrites the turn in
    place — handler is called again, turn_count does not advance, no reset."""
    record = []
    proxy = _make_proxy(_completion_handler(record))

    acc = TokenAccumulator(renderer=None)
    acc.ingest_turn([1, 2, 3], [4, 5])  # turn 1
    acc.update_prefix([{"role": "user", "content": "x"}])

    resp = asyncio.run(
        proxy._handle_cumulative_non_streaming(
            _Request(),
            {"messages": [{"role": "user", "content": "x"}]},
            {"prompt": [1, 2, 3], "add_special_tokens": False, "model": "q"},
            "sess1",
            acc,
            [1, 2, 3],  # same prompt the turn was sampled from
            replay=True,
        )
    )

    assert resp.status_code == 200
    assert record and record[0]["prompt"] == [1, 2, 3]  # regenerated, not cached
    assert acc.turn_count == 1  # overwritten in place, NOT advanced
    assert acc.prev_completion_ids == [91, 92]  # fresh sample replaced the old one
