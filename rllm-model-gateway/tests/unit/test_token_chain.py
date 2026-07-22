"""Tests for linear (delta-chain) token storage.

Covers the write side (``data_process.apply_chain`` + ``build_trace_record`` chain
params), the read side (``token_chain.reconstruct_prompt_ids``), the accumulator
chain pointer, and the end-to-end invariant that reconstruction is byte-identical to
the full prompt while stored data is linear rather than quadratic in turns.
"""

import asyncio

from rllm_model_gateway.data_process import apply_chain, build_trace_record
from rllm_model_gateway.models import TraceRecord
from rllm_model_gateway.store.memory_store import MemoryTraceStore
from rllm_model_gateway.token_accumulator import ResetReason, TokenAccumulator
from rllm_model_gateway.token_chain import reconstruct_prompt_ids


def _resp(prompt_ids, completion_ids):
    """A minimal vLLM-style response body carrying prompt + completion token ids."""
    return {
        "choices": [{"message": {"role": "assistant", "content": "x"}, "token_ids": list(completion_ids), "finish_reason": "stop"}],
        "prompt_token_ids": list(prompt_ids),
    }


# --------------------------------------------------------------------------
# apply_chain (write side)
# --------------------------------------------------------------------------


def test_apply_chain_compresses_extension():
    trace = TraceRecord(trace_id="t1", session_id="s", prompt_token_ids=[1, 2, 3, 4, 5], completion_token_ids=[9])
    apply_chain(trace, parent_len=3, parent_trace_id="t0")
    assert trace.prompt_delta_token_ids == [4, 5]
    assert trace.parent_trace_id == "t0"
    assert trace.prompt_token_ids == []  # full prompt no longer stored


def test_apply_chain_noop_for_root():
    trace = TraceRecord(trace_id="t0", session_id="s", prompt_token_ids=[1, 2, 3], completion_token_ids=[9])
    apply_chain(trace, parent_len=0, parent_trace_id=None)  # turn 0
    assert trace.prompt_delta_token_ids is None
    assert trace.prompt_token_ids == [1, 2, 3]


def test_apply_chain_noop_when_prompt_shorter_than_parent():
    # Duplicate-replay case: prompt is shorter than the accumulated cumulative.
    trace = TraceRecord(trace_id="t2b", session_id="s", prompt_token_ids=[1, 2, 3], completion_token_ids=[9])
    apply_chain(trace, parent_len=5, parent_trace_id="t1")
    assert trace.prompt_delta_token_ids is None  # stored as a full-prompt root
    assert trace.prompt_token_ids == [1, 2, 3]


# --------------------------------------------------------------------------
# reconstruct_prompt_ids (read side)
# --------------------------------------------------------------------------


def _link(trace_id, parent, delta, completion):
    return {
        "trace_id": trace_id,
        "session_id": "s",
        "prompt_token_ids": [],
        "prompt_delta_token_ids": list(delta),
        "parent_trace_id": parent,
        "completion_token_ids": list(completion),
    }


def _root(trace_id, prompt, completion):
    return {
        "trace_id": trace_id,
        "session_id": "s",
        "prompt_token_ids": list(prompt),
        "prompt_delta_token_ids": None,
        "parent_trace_id": None,
        "completion_token_ids": list(completion),
    }


def test_reconstruct_linear_chain():
    traces = [
        _root("t0", [1, 2, 3], [4, 5]),
        _link("t1", "t0", [6, 7], [8]),
        _link("t2", "t1", [9], [10]),
    ]
    reconstruct_prompt_ids(traces)
    assert traces[0]["prompt_token_ids"] == [1, 2, 3]
    assert traces[1]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7]  # cum(t0) + delta
    assert traces[2]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]  # cum(t1) + delta
    # deltas collapsed after reconstruction so the payload carries the full prompt once
    assert all(t["prompt_delta_token_ids"] is None for t in traces)


def test_reconstruct_order_independent():
    # Children before parents in the list: parent-pointer walk still resolves.
    traces = [
        _link("t2", "t1", [9], [10]),
        _link("t1", "t0", [6, 7], [8]),
        _root("t0", [1, 2, 3], [4, 5]),
    ]
    reconstruct_prompt_ids(traces)
    by_id = {t["trace_id"]: t for t in traces}
    assert by_id["t2"]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_reconstruct_segment_break():
    # A mid-session root (post-reset/compaction) starts a fresh chain.
    traces = [
        _root("t0", [1, 2], [3]),
        _link("t1", "t0", [4], [5]),
        _root("t2", [100, 101], [102]),  # segment break
        _link("t3", "t2", [103], [104]),
    ]
    reconstruct_prompt_ids(traces)
    by_id = {t["trace_id"]: t for t in traces}
    assert by_id["t1"]["prompt_token_ids"] == [1, 2, 3, 4]
    assert by_id["t3"]["prompt_token_ids"] == [100, 101, 102, 103]


def test_reconstruct_replay_siblings():
    # Duplicate replay: two children share the same parent; both reconstruct.
    traces = [
        _root("t0", [1, 2], [3]),
        _link("t1", "t0", [4], [5]),
        _link("t1b", "t0", [4], [6]),  # replay sibling
    ]
    reconstruct_prompt_ids(traces)
    by_id = {t["trace_id"]: t for t in traces}
    assert by_id["t1"]["prompt_token_ids"] == [1, 2, 3, 4]
    assert by_id["t1b"]["prompt_token_ids"] == [1, 2, 3, 4]


def test_reconstruct_missing_parent_left_untouched():
    # Partial (since/limit) query: a link whose parent is absent is not fabricated.
    traces = [_link("t1", "t0-absent", [6, 7], [8])]
    reconstruct_prompt_ids(traces)
    assert traces[0]["prompt_delta_token_ids"] == [6, 7]  # unchanged
    assert traces[0]["prompt_token_ids"] == []


def test_reconstruct_legacy_full_prompt_untouched():
    traces = [{"trace_id": "t0", "session_id": "s", "prompt_token_ids": [1, 2, 3], "completion_token_ids": [4]}]
    reconstruct_prompt_ids(traces)
    assert traces[0]["prompt_token_ids"] == [1, 2, 3]


# --------------------------------------------------------------------------
# accumulator chain pointer
# --------------------------------------------------------------------------


def test_accumulator_reset_clears_chain_pointer():
    acc = TokenAccumulator(renderer=None, session_id="s")
    acc.last_trace_id = "t5"
    acc.reset(ResetReason.MANUAL)
    assert acc.last_trace_id is None


# --------------------------------------------------------------------------
# build_trace_record chain params + end-to-end round trip
# --------------------------------------------------------------------------


def test_build_trace_record_delta_roundtrip():
    root_prompt, root_completion = [1, 2, 3], [4, 5]
    root = build_trace_record("s", {}, _resp(root_prompt, root_completion), 0.0, trace_id="t0")
    assert root.prompt_delta_token_ids is None and root.prompt_token_ids == root_prompt

    # Turn 1: prompt extends the root cumulative; delta stored against parent_len.
    parent_len = len(root_prompt) + len(root_completion)
    turn1_prompt = root_prompt + root_completion + [6, 7]  # bridge extension
    turn1 = build_trace_record("s", {}, _resp(turn1_prompt, [8]), 0.0, trace_id="t1", parent_len=parent_len, parent_trace_id="t0")
    assert turn1.prompt_delta_token_ids == [6, 7]
    assert turn1.prompt_token_ids == []

    traces = [root.model_dump(), turn1.model_dump()]
    reconstruct_prompt_ids(traces)
    assert traces[1]["prompt_token_ids"] == turn1_prompt  # byte-identical to the full prompt


def test_storage_is_linear_and_reconstruction_exact():
    """Multi-turn chain: stored token data is linear; reconstruction is byte-exact."""
    n_turns, per_turn = 8, 100
    stored, cumulative, full_prompts = [], [], []
    prev_len = 0
    for k in range(n_turns):
        prompt = cumulative + list(range(1000 + k * per_turn, 1000 + k * per_turn + per_turn))
        completion = [500 + k]
        full_prompts.append(list(prompt))
        if k == 0:
            t = build_trace_record("s", {}, _resp(prompt, completion), 0.0, trace_id=f"t{k}")
        else:
            t = build_trace_record("s", {}, _resp(prompt, completion), 0.0, trace_id=f"t{k}", parent_len=prev_len, parent_trace_id=f"t{k - 1}")
        stored.append(t.model_dump())
        cumulative = prompt + completion
        prev_len = len(cumulative)

    stored_prompt_tokens = sum(len(t["prompt_token_ids"]) + len(t.get("prompt_delta_token_ids") or []) for t in stored)
    full_prompt_tokens = sum(len(p) for p in full_prompts)
    # Quadratic (full) storage would be >> linear (delta) storage for 8 turns.
    assert full_prompt_tokens > 3 * stored_prompt_tokens

    reconstruct_prompt_ids(stored)
    for k in range(n_turns):
        assert stored[k]["prompt_token_ids"] == full_prompts[k]


def test_store_roundtrip_reconstructs():
    """A delta trace persisted to the store reads back with the full prompt."""

    async def _run():
        store = MemoryTraceStore()
        root = build_trace_record("s", {}, _resp([1, 2, 3], [4, 5]), 0.0, trace_id="t0")
        turn1 = build_trace_record("s", {}, _resp([1, 2, 3, 4, 5, 6, 7], [8]), 0.0, trace_id="t1", parent_len=5, parent_trace_id="t0")
        await store.store_trace("t0", "s", root.model_dump())
        await store.store_trace("t1", "s", turn1.model_dump())
        raw = await store.get_session_traces("s")
        # On disk turn 1 holds only its delta, not the full prompt.
        raw_t1 = next(t for t in raw if t["trace_id"] == "t1")
        assert raw_t1["prompt_token_ids"] == [] and raw_t1["prompt_delta_token_ids"] == [6, 7]
        reconstruct_prompt_ids(raw)
        assert next(t for t in raw if t["trace_id"] == "t1")["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7]

    asyncio.run(_run())
