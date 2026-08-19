import json

from rllm_model_gateway.models import TraceDelta, TraceGraph, TraceRecord

from scripts.verify_trace_parity_dump import verify_generation


def _records() -> list[TraceRecord]:
    root = TraceRecord(
        trace_id="root",
        session_id="session",
        model="model",
        messages=[{"role": "user", "content": "one"}],
        prompt_token_ids=[1],
        response_message={"role": "assistant", "content": "two"},
        completion_token_ids=[2],
    )
    child = TraceRecord(
        trace_id="child",
        session_id="session",
        model="model",
        messages=[*root.messages, root.response_message, {"role": "user", "content": "three"}],
        prompt_token_ids=[*root.prompt_token_ids, *root.completion_token_ids, 3],
        response_message={"role": "assistant", "content": "four"},
        completion_token_ids=[4],
    )
    return [root, child]


def _write_dump(generation_dir, records: list[TraceRecord], graph: TraceGraph) -> None:
    generation_dir.mkdir(parents=True)
    (generation_dir.parent / "session.json").write_text(json.dumps({"session_id": "session"}))
    (generation_dir / "raw_trace_records.jsonl").write_text("".join(record.model_dump_json() + "\n" for record in records))
    (generation_dir / "trace_graph.json").write_text(graph.model_dump_json())


def test_verifier_rejects_lossless_but_structurally_wrong_all_root_graph(tmp_path):
    records = _records()
    graph = TraceGraph(
        format="compact",
        version=1,
        deltas=[TraceDelta.against(record, None) for record in records],
    )
    # This proves flatten-only parity would be a false positive.
    assert graph.flatten() == records

    generation_dir = tmp_path / "session-test" / "generation-0000"
    _write_dump(generation_dir, records, graph)
    passed, detail, *_ = verify_generation(generation_dir)

    assert not passed
    assert "parent_trace_id" in detail


def test_verifier_accepts_raw_derived_parent_and_suffixes(tmp_path):
    records = _records()
    graph = TraceGraph(format="compact", version=1, deltas=[])
    for record in records:
        graph.add(record)

    generation_dir = tmp_path / "session-test" / "generation-0000"
    _write_dump(generation_dir, records, graph)
    passed, detail, *_ = verify_generation(generation_dir)

    assert passed, detail


def test_verifier_accepts_canonical_graph_built_from_raw_reasoning_aliases(tmp_path):
    root, child = _records()
    root.response_message = {
        "role": "assistant",
        "content": "",
        "reasoning": "inspect the repository",
        "tool_calls": [],
    }
    child.messages = [
        *root.messages,
        {
            "content": "",
            "role": "assistant",
            "tool_calls": [],
            "reasoning_content": "inspect the repository",
            "provider_specific_fields": {
                "refusal": None,
                "reasoning": "inspect the repository",
            },
        },
        {"role": "user", "content": "three"},
    ]
    raw_root = root.model_dump_json()
    raw_child = child.model_dump_json()
    graph = TraceGraph(format="compact", version=1, deltas=[])
    graph.add(root)
    graph.add(child)

    assert graph.deltas[1].parent_trace_id == root.trace_id
    assert "reasoning" not in graph.deltas[0].response_message
    assert graph.deltas[0].response_message["reasoning_content"] == "inspect the repository"
    assert root.model_dump_json() == raw_root
    assert child.model_dump_json() == raw_child

    generation_dir = tmp_path / "session-test" / "generation-0000"
    _write_dump(generation_dir, [root, child], graph)
    passed, detail, *_ = verify_generation(generation_dir)

    assert passed, detail
