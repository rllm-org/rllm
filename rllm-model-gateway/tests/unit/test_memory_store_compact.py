"""Message compaction in MemoryTraceStore.

Agentic sessions resend the whole conversation each call; the store must keep
each unique (parent, content) message once while every read returns the exact
list the writer stored. The adversarial shapes here are taken from real
DeepSWE runs: strict prefix growth, identical empty messages at different
positions (throttled-call retries), and mid-run history rewrites
(non-cumulative steps).

Tests drive the async store API through ``asyncio.run`` so they need no
pytest-asyncio plugin and run in both the gateway and root test environments.
"""

import asyncio

from rllm_model_gateway.store.memory_store import MemoryTraceStore

_run = asyncio.run


def _trace(messages, completion="ok"):
    return {
        "messages": messages,
        "response_message": {"role": "assistant", "content": completion},
        "prompt_token_ids": [1, 2, 3],
        "completion_token_ids": [4],
    }


def _convo(n):
    """A cumulative conversation: system + n user/assistant pairs."""
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"observation {i}"})
        msgs.append({"role": "assistant", "content": f"action {i}"})
    return msgs


def test_roundtrip_exact_for_cumulative_session():
    store = MemoryTraceStore(compact=True)
    originals = []
    for turn in range(1, 6):
        msgs = _convo(turn)
        originals.append(msgs)
        _run(store.store_trace(f"t{turn}", "s1", _trace(msgs)))

    for turn, msgs in enumerate(originals, start=1):
        assert _run(store.get_trace(f"t{turn}"))["messages"] == msgs

    traces = _run(store.get_session_traces("s1"))
    assert [t["messages"] for t in traces] == originals


def test_compaction_stores_each_unique_message_once():
    store = MemoryTraceStore(compact=True)
    for turn in range(1, 11):
        _run(store.store_trace(f"t{turn}", "s1", _trace(_convo(turn))))
    # 10 turns resend a growing prefix: stored verbatim that is ~110 messages;
    # the node table holds only the 21 unique ones.
    assert len(store._session_nodes["s1"]) == len(_convo(10))


def test_identical_content_at_different_positions_stays_distinct():
    """Repeated empty assistant messages (throttle retries) must not collapse."""
    store = MemoryTraceStore(compact=True)
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "retry"},
        {"role": "assistant", "content": ""},  # same content, different position
    ]
    _run(store.store_trace("t1", "s1", _trace(msgs)))
    assert _run(store.get_trace("t1"))["messages"] == msgs
    assert len(store._session_nodes["s1"]) == 4  # Merkle identity: 4 nodes, not 3


def test_history_rewrite_forks_and_roundtrips():
    """Non-cumulative step: a rewritten prefix forks the chain; both reads stay exact."""
    store = MemoryTraceStore(compact=True)
    a = [{"role": "user", "content": "start"}, {"role": "assistant", "content": "v1"}]
    b = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "COMPACTED"},
        {"role": "user", "content": "next"},
    ]
    _run(store.store_trace("t1", "s1", _trace(a)))
    _run(store.store_trace("t2", "s1", _trace(b)))
    assert _run(store.get_trace("t1"))["messages"] == a
    assert _run(store.get_trace("t2"))["messages"] == b
    # shared "start" root, then forked tails: 1 + 1 + 2
    assert len(store._session_nodes["s1"]) == 4


def test_sessions_do_not_share_nodes_and_delete_frees_them():
    store = MemoryTraceStore(compact=True)
    msgs = _convo(3)
    _run(store.store_trace("t1", "s1", _trace(msgs)))
    _run(store.store_trace("t2", "s2", _trace(msgs)))
    assert len(store._session_nodes["s1"]) == len(msgs)
    assert len(store._session_nodes["s2"]) == len(msgs)

    _run(store.delete_session("s1"))
    assert "s1" not in store._session_nodes
    assert "s1" not in store._session_node_lookup
    # s2 unaffected
    assert _run(store.get_trace("t2"))["messages"] == msgs


def test_empty_and_nonlist_messages_stored_verbatim():
    store = MemoryTraceStore(compact=True)
    for tid, messages in (("t1", []), ("t2", None), ("t3", "raw-prompt")):
        data = _trace([])
        data["messages"] = messages
        _run(store.store_trace(tid, "s1", data))
        assert _run(store.get_trace(tid))["messages"] == messages


def test_compaction_composes_with_token_id_packing():
    """Both store-side transforms apply; reads restore both."""
    store = MemoryTraceStore(compact=True)
    msgs = _convo(2)
    _run(store.store_trace("t1", "s1", _trace(msgs)))
    got = _run(store.get_trace("t1"))
    assert got["messages"] == msgs
    assert got["prompt_token_ids"] == [1, 2, 3]  # unpacked back to a plain list
    assert isinstance(got["prompt_token_ids"], list)


def test_default_mode_stores_verbatim_and_builds_no_tables():
    """Without compact=True the store behaves exactly as before this change."""
    store = MemoryTraceStore()
    msgs = _convo(3)
    _run(store.store_trace("t1", "s1", _trace(msgs)))
    assert _run(store.get_trace("t1"))["messages"] == msgs
    assert len(store._session_nodes) == 0
    assert len(store._session_node_lookup) == 0
    # compactal representation is the verbatim list, not a leaf marker
    assert isinstance(store._traces["t1"]["messages"], list)


def _trace_ids(messages, prompt_ids):
    d = _trace(messages)
    d["prompt_token_ids"] = prompt_ids
    return d


def test_prompt_ids_delta_roundtrip_and_dedup():
    """Prompt token ids repeat the previous call's — compact stores suffixes only."""
    store = MemoryTraceStore(compact=True)
    full = []
    originals = []
    for turn in range(1, 6):
        full = full + list(range((turn - 1) * 100, turn * 100))
        originals.append(list(full))
        _run(store.store_trace(f"t{turn}", "s1", _trace_ids(_convo(turn), list(full))))
    # every read reconstructs the full ids
    for turn, ids in enumerate(originals, start=1):
        assert _run(store.get_trace(f"t{turn}"))["prompt_token_ids"] == ids
    bulk = _run(store.get_session_traces("s1"))
    assert [t["prompt_token_ids"] for t in bulk] == originals
    # storage is linear: traces 2..5 hold only 100-id suffixes
    for turn in range(2, 6):
        marker = store._traces[f"t{turn}"]["prompt_token_ids"]
        assert isinstance(marker, dict)
        _, lcp, suffix = marker["__prompt_ids_delta__"]
        assert lcp == (turn - 1) * 100 and len(suffix) == 100


def test_prompt_ids_non_prefix_falls_back_verbatim():
    """A rewritten prompt (no shared prefix) stores verbatim — lossless always."""
    store = MemoryTraceStore(compact=True)
    _run(store.store_trace("t1", "s1", _trace_ids(_convo(1), [1, 2, 3])))
    _run(store.store_trace("t2", "s1", _trace_ids(_convo(2), [9, 8, 7, 6])))
    assert _run(store.get_trace("t1"))["prompt_token_ids"] == [1, 2, 3]
    assert _run(store.get_trace("t2"))["prompt_token_ids"] == [9, 8, 7, 6]


def test_restore_same_trace_id_does_not_self_chain():
    store = MemoryTraceStore(compact=True)
    _run(store.store_trace("t1", "s1", _trace_ids(_convo(1), [1, 2, 3])))
    _run(store.store_trace("t1", "s1", _trace_ids(_convo(1), [1, 2, 3, 4])))  # retry re-store
    assert _run(store.get_trace("t1"))["prompt_token_ids"] == [1, 2, 3, 4]


def test_delete_session_rematerializes_shared_traces():
    """Audit repro: trace stored under two sessions must survive either deletion."""
    store = MemoryTraceStore(compact=True)
    msgs = _convo(2)
    _run(store.store_trace("shared", "s1", _trace_ids(msgs, [1, 2, 3])))
    _run(store.store_trace("shared", "s2", _trace_ids(msgs, [1, 2, 3])))  # re-store repoints markers at s2
    _run(store.delete_session("s2"))
    got = _run(store.get_trace("shared"))  # must not KeyError into s2's dropped tables
    assert got["messages"] == msgs
    assert got["prompt_token_ids"] == [1, 2, 3]
    traces = _run(store.get_session_traces("s1"))
    assert traces and traces[0]["messages"] == msgs


def test_compact_fetch_shares_node_objects_and_matches_default():
    """Compact fetch expands to the same content; shared prefixes share dicts."""
    from rllm_model_gateway.client import _expand_compact_traces

    for compact in (True, False):
        store = MemoryTraceStore(compact=compact)
        for turn in range(1, 4):
            _run(store.store_trace(f"t{turn}", "s1", _trace(_convo(turn))))
        default = [t["messages"] for t in _run(store.get_session_traces("s1"))]
        payload = _run(store.get_session_traces_compact("s1"))
        expanded = [t["messages"] for t in _expand_compact_traces(payload)]
        assert expanded == default
        # prefix sharing: trace 2's first message IS trace 1's first message
        assert expanded[1][0] is expanded[0][0]
        assert expanded[2][0] is expanded[0][0]


def test_compact_fetch_ships_prompt_id_deltas_and_expands():
    """Wire carries id suffixes, not full ids; client rebuilds chains exactly."""
    import json as _json

    from rllm_model_gateway.client import _expand_compact_traces

    store = MemoryTraceStore(compact=True)
    full = []
    originals = []
    for turn in range(1, 5):
        full = full + list(range((turn - 1) * 50, turn * 50))
        originals.append(list(full))
        _run(store.store_trace(f"t{turn}", "s1", _trace_ids(_convo(turn), list(full))))
    payload = _json.loads(_json.dumps(_run(store.get_session_traces_compact("s1"))))
    # deltas on the wire: traces 2..4 ship 50-id suffixes
    assert all("prompt_ids_delta" in t for t in payload["traces"][1:])
    assert all(len(t["prompt_ids_delta"][2]) == 50 for t in payload["traces"][1:])
    expanded = _expand_compact_traces(payload)
    assert [t["prompt_token_ids"] for t in expanded] == originals
    assert all("_tid" not in t for t in expanded)


def test_compact_fetch_resolves_markers_across_sessions():
    """Audit repro at the fetch layer: marker owned by another session must resolve."""
    from rllm_model_gateway.client import _expand_compact_traces

    store = MemoryTraceStore(compact=True)
    msgs = _convo(2)
    _run(store.store_trace("shared", "s1", _trace(msgs)))
    _run(store.store_trace("shared", "s2", _trace(msgs)))  # marker now points at s2's table
    payload = _run(store.get_session_traces_compact("s1"))  # fetch via s1 must not KeyError
    expanded = _expand_compact_traces(payload)
    assert expanded and expanded[0]["messages"] == msgs


def test_replay_restore_cannot_form_delta_cycle():
    """Review repro: t1, t2(delta->t1), re-store t1 must not create t1->t2->t1."""
    store = MemoryTraceStore(compact=True)
    _run(store.store_trace("t1", "s", {"messages": [{"role": "user", "content": "a"}], "prompt_token_ids": [1, 2, 3]}))
    _run(store.store_trace("t2", "s", {"messages": [{"role": "user", "content": "b"}], "prompt_token_ids": [1, 2, 3, 4]}))
    _run(store.store_trace("t1", "s", {"messages": [{"role": "user", "content": "a"}], "prompt_token_ids": [1, 2, 3, 4, 5]}))
    # t2 must still reconstruct against t1's OLD content; t1 returns its new ids
    assert _run(store.get_trace("t2"))["prompt_token_ids"] == [1, 2, 3, 4]
    assert _run(store.get_trace("t1"))["prompt_token_ids"] == [1, 2, 3, 4, 5]
    bulk = _run(store.get_session_traces("s"))
    assert [t["prompt_token_ids"] for t in bulk] == [[1, 2, 3, 4, 5], [1, 2, 3, 4]]


def test_interleaved_lineages_anchor_independently():
    store = MemoryTraceStore(compact=True)
    _run(store.store_trace("p1", "s", {"messages": [{"role": "user", "content": "p"}], "prompt_token_ids": [1, 2], "lineage_id": "parent"}))
    _run(store.store_trace("c1", "s", {"messages": [{"role": "user", "content": "c"}], "prompt_token_ids": [9, 8], "lineage_id": "child"}))
    _run(store.store_trace("p2", "s", {"messages": [{"role": "user", "content": "p2"}], "prompt_token_ids": [1, 2, 3], "lineage_id": "parent"}))
    # p2 deltas against p1 (its lineage), not against c1
    marker = store._traces["p2"]["prompt_token_ids"]
    assert isinstance(marker, dict) and marker["__prompt_ids_delta__"][0] == "p1"
    assert _run(store.get_trace("p2"))["prompt_token_ids"] == [1, 2, 3]


def test_since_filter_rebases_orphaned_deltas():
    """Review repro: excluding a delta ancestor must yield a self-contained payload."""
    import json as _json

    from rllm_model_gateway.client import _expand_compact_traces

    store = MemoryTraceStore(compact=True)
    for i, ids in enumerate(([1, 2], [1, 2, 3], [1, 2, 3, 4])):
        _run(store.store_trace(f"t{i}", "s", {"messages": [{"role": "user", "content": str(i)}], "prompt_token_ids": ids}))
    store._timestamps["t0"] = 1.0
    store._timestamps["t1"] = 2.0
    store._timestamps["t2"] = 3.0
    payload = _json.loads(_json.dumps(_run(store.get_session_traces_compact("s", since=1.5)), default=str))
    out = _expand_compact_traces(payload)
    assert [t["prompt_token_ids"] for t in out] == [[1, 2, 3], [1, 2, 3, 4]]


def test_cycle_guard_raises_instead_of_hanging():
    store = MemoryTraceStore(compact=True)
    # forge a corrupt two-node cycle directly
    store._traces["a"] = {"prompt_token_ids": {"__prompt_ids_delta__": ["b", 1, [2]]}}
    store._traces["b"] = {"prompt_token_ids": {"__prompt_ids_delta__": ["a", 1, [3]]}}
    store._session_index["s"] = ["a", "b"]
    import pytest as _pytest

    with _pytest.raises(ValueError, match="cycle"):
        _run(store.get_trace("a"))


def test_delete_then_reuse_session_never_deltas_against_deleted_traces():
    """Review F1: tuple-keyed anchors must die with their session."""
    store = MemoryTraceStore(compact=True)
    _run(store.store_trace("a1", "sess", {"messages": [{"role": "user", "content": "x"}], "prompt_token_ids": list(range(1013))}))
    _run(store.delete_session("sess"))
    assert len(store._session_last_prompt_ids) == 0  # no anchor leak
    _run(store.store_trace("b1", "sess", {"messages": [{"role": "user", "content": "y"}], "prompt_token_ids": list(range(1701))}))
    assert _run(store.get_trace("b1"))["prompt_token_ids"] == list(range(1701))


def test_cross_session_trace_id_overwrite_rebases_all_dependents():
    """Review F2: dependents in OTHER sessions must be pinned before overwrite."""
    store = MemoryTraceStore(compact=True)
    _run(store.store_trace("shared", "s1", {"messages": [{"role": "user", "content": "a"}], "prompt_token_ids": list(range(1013))}))
    _run(store.store_trace("dep", "s1", {"messages": [{"role": "user", "content": "b"}], "prompt_token_ids": list(range(1701))}))
    _run(store.store_trace("shared", "s2", {"messages": [{"role": "user", "content": "c"}], "prompt_token_ids": list(range(997))}))
    assert _run(store.get_trace("dep"))["prompt_token_ids"] == list(range(1701))
    assert _run(store.get_trace("shared"))["prompt_token_ids"] == list(range(997))


def test_stale_anchor_in_other_session_cannot_mint_corrupt_deltas():
    """Review P1a repro: silent [9,9,x] corruption via a cross-session overwrite."""
    store = MemoryTraceStore(compact=True)
    _run(store.store_trace("shared", "s1", {"messages": [{"role": "u", "content": "a"}], "prompt_token_ids": [1, 2, 3]}))
    _run(store.store_trace("shared", "s2", {"messages": [{"role": "u", "content": "b"}], "prompt_token_ids": [9, 9, 9]}))
    _run(store.store_trace("next", "s1", {"messages": [{"role": "u", "content": "c"}], "prompt_token_ids": [1, 2, 3, 4]}))
    assert _run(store.get_trace("next"))["prompt_token_ids"] == [1, 2, 3, 4]


def test_message_walk_guards_cycles_and_dangling():
    """Review P2f: forged cycles must raise, not hang the event loop."""
    import pytest as _pytest

    store = MemoryTraceStore(compact=True)
    store._session_nodes["s"] = {"a": ("b", {"role": "u", "content": "1"}), "b": ("a", {"role": "u", "content": "2"})}
    store._traces["t"] = {"messages": {"__messages_ref__": ["s", "a", 2]}}
    store._session_index["s"] = ["t"]
    with _pytest.raises(ValueError, match="cycle"):
        _run(store.get_trace("t"))
    store._traces["t2"] = {"messages": {"__messages_ref__": ["s", "missing", 1]}}
    store._session_index["s"].append("t2")
    with _pytest.raises(ValueError, match="dangling"):
        _run(store.get_trace("t2"))
