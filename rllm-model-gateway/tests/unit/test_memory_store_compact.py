"""Focused tests for the graph-backed in-memory trace store."""

import asyncio
import json

import pytest
from rllm_model_gateway.models import TraceGraph, TraceRecord
from rllm_model_gateway.store.memory_store import MemoryTraceStore

_run = asyncio.run
_HEAVY_FIELDS = ("messages", "prompt_token_ids", "response_message", "completion_token_ids")
_REQUIRED_FIELDS = ("messages", "response_message")


def _trajectory(turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u0"}]
    prompt_ids = [1, 10]
    records = []
    for i in range(turns):
        response = {"role": "assistant", "content": f"a{i}"}
        completion = [100 + i]
        records.append(
            {
                "lineage_id": "main",
                "messages": list(messages),
                "prompt_token_ids": list(prompt_ids),
                "response_message": response,
                "completion_token_ids": completion,
            }
        )
        messages += [response, {"role": "user", "content": f"u{i + 1}"}]
        prompt_ids += completion + [20 + i]
    return records


def _core(record: dict) -> dict:
    return {field: record[field] for field in _HEAVY_FIELDS}


def test_compact_roundtrip_and_wire_are_exact_and_linear():
    store = MemoryTraceStore(compact=True)
    records = _trajectory(5)
    for i, record in enumerate(records):
        _run(store.store_trace(f"t{i}", "s", record))

    assert [_core(record) for record in _run(store.get_session_traces("s"))] == [_core(record) for record in records]

    payload = json.loads(json.dumps(_run(store.get_session_traces_compact("s"))))
    graph = TraceGraph.model_validate(payload)
    assert [_core(record.model_dump()) for record in graph.flatten()] == [_core(record) for record in records]
    assert sum(len(delta.messages_suffix) for delta in graph.deltas) == 6
    assert sum(len(delta.prompt_ids_suffix) for delta in graph.deltas) == 6


def test_retry_replaces_the_same_leaf_trace():
    store = MemoryTraceStore(compact=True)
    root, child = _trajectory(2)
    retry = {**child, "completion_token_ids": [999]}
    _run(store.store_trace("root", "s", root))
    _run(store.store_trace("child", "s", child))
    _run(store.store_trace("child", "s", retry))

    deltas = {delta.trace_id: delta for delta in store._graphs["s"].deltas}
    assert deltas["child"].parent_trace_id == "root"
    assert [record["trace_id"] for record in _run(store.get_session_traces("s"))] == ["root", "child"]
    assert _run(store.get_trace("child"))["completion_token_ids"] == [999]


def test_retry_cannot_replace_a_trace_with_children():
    store = MemoryTraceStore(compact=True)
    root, child = _trajectory(2)
    _run(store.store_trace("root", "s", root))
    _run(store.store_trace("child", "s", child))

    with pytest.raises(ValueError, match="after it has children"):
        _run(store.store_trace("root", "s", root))


@pytest.mark.parametrize("field", _REQUIRED_FIELDS)
def test_compact_fields_are_required(field):
    store = MemoryTraceStore(compact=True)
    record = _trajectory(1)[0]
    del record[field]
    with pytest.raises(ValueError, match=field):
        _run(store.store_trace("t", "s", record))


def test_missing_token_ids_are_normalized_to_empty_lists():
    store = MemoryTraceStore(compact=True)
    record = _trajectory(1)[0]
    del record["prompt_token_ids"], record["completion_token_ids"]

    _run(store.store_trace("t", "s", record))

    restored = _run(store.get_trace("t"))
    assert restored["prompt_token_ids"] == []
    assert restored["completion_token_ids"] == []


def test_sessions_own_their_graphs_and_delete_independently():
    store = MemoryTraceStore(compact=True)
    first, second = _trajectory(2)
    _run(store.store_trace("a", "s1", first))
    _run(store.store_trace("b", "s2", second))

    assert set(store._graphs) == {"s1", "s2"}
    assert _run(store.delete_session("s1")) == 1
    assert _run(store.get_trace("a")) is None
    assert _core(_run(store.get_trace("b"))) == _core(second)


def test_since_slice_is_self_contained():
    store = MemoryTraceStore(compact=True)
    records = _trajectory(3)
    for i, record in enumerate(records):
        _run(store.store_trace(f"t{i}", "s", record))
        store._timestamps[f"t{i}"] = float(i)

    payload = _run(store.get_session_traces_compact("s", since=1.0))
    flat = TraceGraph.model_validate(payload).flatten()
    assert [record.trace_id for record in flat] == ["t1", "t2"]
    assert [_core(record.model_dump()) for record in flat] == [_core(record) for record in records[1:]]


def test_count_does_not_resolve(monkeypatch):
    store = MemoryTraceStore(compact=True)
    for i, record in enumerate(_trajectory(3)):
        _run(store.store_trace(f"t{i}", "s", record))

    monkeypatch.setattr(TraceGraph, "resolve", lambda *_: pytest.fail("resolve called"))
    assert _run(store.count_session_traces("s")) == 3


def test_default_mode_remains_verbatim():
    store = MemoryTraceStore()
    record = {**_trajectory(1)[0], "extra": object()}
    _run(store.store_trace("t", "s", record))
    assert _run(store.get_trace("t")) == record
    assert not store._graphs


def test_raw_capture_is_rejected_for_compact_store():
    from rllm_model_gateway.proxy import ReverseProxy

    with pytest.raises(ValueError, match="capture_raw_payloads"):
        ReverseProxy(router=None, store=MemoryTraceStore(compact=True), capture_raw_payloads=True)


def test_trace_parity_dump_preserves_raw_inputs_and_eventual_graph(tmp_path):
    store = MemoryTraceStore(compact=True, trace_parity_dump_dir=str(tmp_path))
    root, child = _trajectory(2)
    child_retry = {**child, "completion_token_ids": [999]}

    _run(store.store_trace("root", "session/one", root))
    _run(store.store_trace("child", "session/one", child))
    _run(store.store_trace("child", "session/one", child_retry))
    _run(store.get_session_traces_compact("session/one"))

    session_dir = next(tmp_path.glob("session-*"))
    assert json.loads((session_dir / "session.json").read_text()) == {"session_id": "session/one"}
    generation_zero = session_dir / "generation-0000"
    raw = [TraceRecord.model_validate_json(line) for line in (generation_zero / "raw_trace_records.jsonl").read_text().splitlines()]
    graph = TraceGraph.model_validate_json((generation_zero / "trace_graph.json").read_text())

    assert [record.trace_id for record in raw] == ["root", "child", "child"]
    assert graph.flatten() == [raw[0], raw[2]]

    # Deleting and recreating a session id is how rollout retries are isolated.
    _run(store.delete_session("session/one"))
    _run(store.store_trace("new-root", "session/one", root))
    _run(store.get_session_traces_compact("session/one"))
    assert (session_dir / "generation-0001" / "raw_trace_records.jsonl").exists()
    assert (session_dir / "generation-0001" / "trace_graph.json").exists()


def test_raw_trace_record_is_dumped_before_graph_conversion(tmp_path, monkeypatch):
    store = MemoryTraceStore(compact=True, trace_parity_dump_dir=str(tmp_path))
    record = _trajectory(1)[0]
    original_add = TraceGraph.add

    def assert_raw_dump_exists_before_add(graph, raw_record):
        raw_path = next(tmp_path.glob("session-*/generation-0000/raw_trace_records.jsonl"))
        dumped = TraceRecord.model_validate_json(raw_path.read_text().strip())
        assert dumped == raw_record
        return original_add(graph, raw_record)

    monkeypatch.setattr(TraceGraph, "add", assert_raw_dump_exists_before_add)
    _run(store.store_trace("raw", "session", record))
