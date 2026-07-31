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


def _make_proxy(local_handler):
    """ReverseProxy wired for the local cumulative path (no router/worker)."""
    return ReverseProxy(
        router=None,
        store=MemoryTraceStore(),
        sync_traces=True,
        local_handler=local_handler,
        cumulative_token_mode=True,
        renderer=None,  # accumulator state is set directly in these tests
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


# ----------------------------------------------------------------------------
# R3 router replay over the cumulative-token path.
#
# The token-in path (_token_prompt_completion) must surface the TokenOutput's
# per-token routing matrices on the choice, and the proxy must carry them onto
# the persisted trace. Without this, router replay silently no-ops whenever
# cumulative_token_mode is on (turns 2+ all take the token-in path).
# ----------------------------------------------------------------------------


def _routing_engine(routing_matrices):
    """Fake TinkerEngine token-in path: TokenOutput carries routing_matrices,
    but assemble_model_output does NOT copy them onto the ModelOutput (mirrors
    the real TinkerEngine — only the get_model_response* wrappers copy them)."""
    from types import SimpleNamespace

    class _Engine:
        model_name = "q"

        async def get_token_output_from_token_input(self, prompt_ids, **kwargs):
            return SimpleNamespace(routing_matrices=routing_matrices)

        def assemble_model_output(self, prompt_ids, token_output):
            return SimpleNamespace(
                content="next action",
                text="next action",
                prompt_ids=list(prompt_ids),
                completion_ids=[91, 92],
                logprobs=[-0.1, -0.2],
                finish_reason="stop",
                prompt_length=len(prompt_ids),
                completion_length=2,
                weight_version=7,
            )

    return _Engine()


def test_token_prompt_completion_carries_routing_matrices():
    from rllm.gateway.tinker_adapter import _token_prompt_completion

    rm = ["layer-blob-a", "layer-blob-b"]
    resp = asyncio.run(_token_prompt_completion(_routing_engine(rm), {"model": "q"}, [1, 2, 3], {}))
    # Read off TokenOutput, not ModelOutput (which never carried them here).
    # Stamped under vLLM's key so the gateway has one field to read.
    assert resp["choices"][0]["routed_experts"] == rm


def test_token_prompt_completion_routing_matrices_none_when_absent():
    """Dense models / R3 disabled: no matrices → field is None, not an error."""
    from rllm.gateway.tinker_adapter import _token_prompt_completion

    resp = asyncio.run(_token_prompt_completion(_routing_engine(None), {"model": "q"}, [1, 2, 3], {}))
    assert resp["choices"][0]["routed_experts"] is None


def test_cumulative_local_persists_routing_matrices_to_trace():
    """End-to-end over the proxy: a handler emitting per-token routing matrices
    (as the fixed token-in path now does) must land them on the persisted trace
    so training can replay expert routing."""
    rm = ["blob-a", "blob-b"]

    async def handler(body):
        return {
            "id": "cmpl-x",
            "object": "text_completion",
            "choices": [
                {
                    "index": 0,
                    "text": "next action",
                    "token_ids": [91, 92],
                    "finish_reason": "stop",
                    "routed_experts": rm,
                    "logprobs": {"token_logprobs": [-0.1, -0.2]},
                }
            ],
            "prompt_token_ids": body["prompt"],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 2},
        }

    store = MemoryTraceStore()
    proxy = ReverseProxy(
        router=None,
        store=store,
        sync_traces=True,
        local_handler=handler,
        cumulative_token_mode=True,
        renderer=None,
    )
    acc = TokenAccumulator(renderer=None)
    acc.ingest_turn([1, 2, 3], [4, 5])
    bridged = [1, 2, 3, 4, 5, 6, 7]

    asyncio.run(
        proxy._handle_cumulative_non_streaming(
            _Request(),
            {"messages": [{"role": "user", "content": "x"}]},
            {"prompt": bridged, "add_special_tokens": False, "model": "q"},
            "sess1",
            acc,
            bridged,
        )
    )

    traces = asyncio.run(store.get_session_traces("sess1"))
    assert traces and traces[0]["routing_matrices"] == rm
