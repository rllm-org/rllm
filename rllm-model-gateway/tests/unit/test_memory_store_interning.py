"""Message interning in MemoryTraceStore.

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


def test_interning_stores_each_unique_message_once():
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
    assert "s1" not in store._session_chain
    # s2 unaffected
    assert _run(store.get_trace("t2"))["messages"] == msgs


def test_empty_and_nonlist_messages_stored_verbatim():
    store = MemoryTraceStore(compact=True)
    for tid, messages in (("t1", []), ("t2", None), ("t3", "raw-prompt")):
        data = _trace([])
        data["messages"] = messages
        _run(store.store_trace(tid, "s1", data))
        assert _run(store.get_trace(tid))["messages"] == messages


def test_interning_composes_with_token_id_packing():
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
    assert len(store._session_chain) == 0
    # internal representation is the verbatim list, not a leaf marker
    assert isinstance(store._traces["t1"]["messages"], list)
