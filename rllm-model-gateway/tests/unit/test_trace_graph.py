"""Tests for the minimal TraceDelta/TraceGraph contract."""

import json

import pytest
from pydantic import ValidationError
from rllm_model_gateway.models import TraceDelta, TraceGraph, TraceRecord


def _record(
    i: int,
    messages: list[dict],
    prompt_ids: list[int],
    *,
    session: str = "s",
    lineage: str | None = None,
    trace_id: str | None = None,
) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id or f"t{i}",
        session_id=session,
        lineage_id=lineage,
        model="model",
        messages=messages,
        prompt_token_ids=prompt_ids,
        response_message={"role": "assistant", "content": f"a{i}"},
        completion_token_ids=[900 + i],
        logprobs=[-0.1 * (i + 1)],
        finish_reason="stop",
        token_counts={"prompt": len(prompt_ids), "completion": 1},
        timestamp=float(i),
        metadata={"turn": i},
    )


def _conversation(turns: int, *, lineage: str | None = None, base: int = 0) -> list[TraceRecord]:
    records = []
    messages = [{"role": "system", "content": f"system-{base}"}]
    prompt_ids = [base, base + 1]
    for i in range(turns):
        messages = [*messages, {"role": "user", "content": f"u{base}.{i}"}]
        prompt_ids = [*prompt_ids, base + 10 + i]
        record = _record(
            i,
            list(messages),
            list(prompt_ids),
            lineage=lineage,
            trace_id=f"t{base}.{i}",
        )
        records.append(record)
        messages = [*messages, record.response_message]
        prompt_ids = [*prompt_ids, *record.completion_token_ids]
    return records


def _graph(records: list[TraceRecord] | None = None) -> TraceGraph:
    graph = TraceGraph(format="compact", version=1, deltas=[])
    for record in records or []:
        graph.add(record)
    return graph


def _assert_exact(actual: TraceRecord, expected: TraceRecord) -> None:
    assert actual.model_dump(mode="json") == expected.model_dump(mode="json")


def _direct_delta(
    trace_id: str,
    *,
    parent: str | None = None,
    session: str = "s",
    lineage: str | None = None,
) -> TraceDelta:
    return TraceDelta(
        trace_id=trace_id,
        session_id=session,
        lineage_id=lineage,
        parent_trace_id=parent,
        model="model",
        messages_suffix=[{"role": "user", "content": trace_id}],
        prompt_ids_suffix=[1],
        response_message={"role": "assistant", "content": f"a-{trace_id}"},
        completion_token_ids=[2],
    )


@pytest.mark.parametrize("missing", ["format", "version", "deltas"])
def test_graph_contract_fields_are_required(missing: str):
    payload = {"format": "compact", "version": 1, "deltas": []}
    payload.pop(missing)
    with pytest.raises(ValidationError):
        TraceGraph.model_validate(payload)


def test_graph_accepts_only_version_one():
    with pytest.raises(ValidationError):
        TraceGraph.model_validate({"format": "compact", "version": 2, "deltas": []})


@pytest.mark.parametrize(
    "missing",
    [
        "trace_id",
        "session_id",
        "parent_trace_id",
        "messages_suffix",
        "prompt_ids_suffix",
        "response_message",
        "completion_token_ids",
    ],
)
def test_delta_core_fields_are_required(missing: str):
    payload = _direct_delta("t").model_dump()
    payload.pop(missing)
    with pytest.raises(ValidationError):
        TraceDelta.model_validate(payload)


def test_root_and_child_deltas_resolve_exactly():
    parent = _record(0, [{"role": "user", "content": "question"}], [1, 2])

    root = TraceDelta.against(parent, None)
    assert root.parent_trace_id is None
    _assert_exact(TraceGraph(format="compact", version=1, deltas=[root]).resolve(parent.trace_id), parent)

    continuation = _record(
        2,
        [*parent.messages, parent.response_message, {"role": "user", "content": "next"}],
        [*parent.prompt_token_ids, *parent.completion_token_ids, 3],
    )
    child = TraceDelta.against(continuation, parent)
    assert child.parent_trace_id == parent.trace_id

    graph = TraceGraph(format="compact", version=1, deltas=[root, child])
    _assert_exact(graph.resolve(continuation.trace_id), continuation)


def test_message_identity_preserves_key_order_for_byte_parity():
    parent = _record(0, [{"role": "user", "content": {"a": 1, "b": 2}}], [1])
    same_message = {"content": {"b": 2, "a": 1}, "role": "user"}
    child = _record(1, [same_message, {"role": "user", "content": "next"}], [1, 2])

    graph = _graph([parent, child])

    assert graph.deltas[1].parent_trace_id is None
    _assert_exact(graph.resolve(child.trace_id), child)


@pytest.mark.parametrize("facet", ["messages", "tokens"])
def test_non_prefix_input_becomes_a_root(facet: str):
    parent = _record(0, [{"role": "user", "content": "same"}], [1, 2])
    messages = [*parent.messages, {"role": "user", "content": "next"}]
    prompt_ids = [*parent.prompt_token_ids, 3]
    if facet == "messages":
        messages[0] = {"role": "user", "content": "different"}
    else:
        prompt_ids[0] = 99
    child = _record(1, messages, prompt_ids)

    delta = TraceDelta.against(child, parent)

    assert delta.parent_trace_id is None
    _assert_exact(TraceGraph(format="compact", version=1, deltas=[delta]).resolve(child.trace_id), child)


def test_append_rejects_duplicate_and_missing_or_forward_parent():
    graph = _graph()
    root = _direct_delta("root")
    graph.append(root)

    with pytest.raises(ValueError, match="duplicate trace id"):
        graph.append(root)
    with pytest.raises(ValueError, match="parent is not earlier"):
        graph.append(_direct_delta("child", parent="future"))

    payload = {
        "format": "compact",
        "version": 1,
        "deltas": [
            _direct_delta("child", parent="parent").model_dump(),
            _direct_delta("parent").model_dump(),
        ],
    }
    with pytest.raises(ValidationError, match="parent is not earlier"):
        TraceGraph.model_validate(payload)


def test_invalid_message_does_not_partially_append():
    graph = _graph()
    delta = _direct_delta("bad")
    delta.response_message["content"] = b"not-json"

    with pytest.raises(TypeError):
        graph.append(delta)
    assert graph.deltas == []


def test_append_rejects_cross_session_and_cross_lineage_edges():
    graph = _graph()
    graph.append(_direct_delta("parent", lineage="one"))

    with pytest.raises(ValueError, match="another session"):
        graph.append(_direct_delta("session-child", session="other"))
    with pytest.raises(ValueError, match="another session or lineage"):
        graph.append(_direct_delta("lineage-child", parent="parent", lineage="two"))


def test_graph_wire_roundtrip_preserves_every_record():
    records = _conversation(8, lineage="main")
    graph = _graph(records)

    wire = json.loads(json.dumps(graph.model_dump(mode="json")))
    restored = TraceGraph.model_validate(wire)

    assert restored.model_dump(mode="json") == graph.model_dump(mode="json")
    for actual, expected in zip(restored.flatten(), records, strict=True):
        _assert_exact(actual, expected)


def test_interleaved_lineages_chain_independently():
    left = _conversation(5, lineage="left", base=100)
    right = _conversation(5, lineage="right", base=200)
    records = [record for pair in zip(left, right, strict=True) for record in pair]
    graph = _graph(records)

    for delta in graph.deltas:
        if delta.parent_trace_id is not None:
            parent = graph.delta(delta.parent_trace_id)
            assert parent is not None
            assert parent.lineage_id == delta.lineage_id
    for actual, expected in zip(graph.flatten(), records, strict=True):
        _assert_exact(actual, expected)


def test_slice_is_ordered_and_self_contained():
    records = _conversation(4)
    graph = _graph(records)

    sliced = graph.slice([records[1].trace_id, "missing", records[2].trace_id])

    assert [delta.trace_id for delta in sliced.deltas] == [records[1].trace_id, records[2].trace_id]
    assert sliced.deltas[0].parent_trace_id is None
    assert sliced.deltas[1].parent_trace_id == records[1].trace_id
    restored = TraceGraph.model_validate(json.loads(json.dumps(sliced.model_dump(mode="json"))))
    for actual, expected in zip(restored.flatten(), records[1:3], strict=True):
        _assert_exact(actual, expected)


def test_deep_chain_resolves_iteratively():
    depth = 1_500
    graph = _graph()
    graph.append(_direct_delta("d0"))
    for i in range(1, depth):
        graph.append(_direct_delta(f"d{i}", parent=f"d{i - 1}"))

    record = graph.resolve(f"d{depth - 1}")

    assert len(record.messages) == 2 * depth - 1
    assert len(record.prompt_token_ids) == 2 * depth - 1
    assert record.messages[-1] == {"role": "user", "content": f"d{depth - 1}"}


def test_retry_replaces_a_leaf_in_place():
    root, child = _conversation(2)
    graph = _graph([root, child])
    retry = child.model_copy(deep=True)
    retry.completion_token_ids = [999]

    graph.replace_leaf(retry)

    assert len(graph.deltas) == 2
    assert graph.deltas[1].parent_trace_id == root.trace_id
    _assert_exact(graph.resolve(child.trace_id), retry)


def test_failed_leaf_replacement_is_atomic():
    record = _record(0, [{"role": "user", "content": "x"}], [1])
    graph = _graph([record])
    replacement = record.model_copy(update={"session_id": "other"})

    with pytest.raises(ValueError, match="another session or lineage"):
        graph.replace_leaf(replacement)
    _assert_exact(graph.resolve(record.trace_id), record)


@pytest.mark.parametrize("field", ["raw_request", "raw_response"])
def test_raw_capture_cannot_be_delta_stored(field: str):
    record = _record(0, [{"role": "user", "content": "x"}], [1])
    setattr(record, field, {"captured": True})

    with pytest.raises(ValueError, match="raw_request/raw_response"):
        TraceDelta.against(record, None)
