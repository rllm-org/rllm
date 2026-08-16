"""Opt-in parity replay over real token-bearing episode artifacts."""

import asyncio
import glob
import json
import os

import pytest
from rllm_model_gateway.models import TraceGraph, TraceRecord
from rllm_model_gateway.store.memory_store import MemoryTraceStore

_run = asyncio.run


def _paths() -> list[str]:
    raw = os.environ.get("RLLM_TOKEN_PARITY_EPISODES")
    if not raw:
        return []
    paths: list[str] = []
    for item in raw.split(os.pathsep):
        paths.extend(sorted(glob.glob(os.path.join(item, "*.json"))) if os.path.isdir(item) else [item])
    limit = int(os.environ.get("RLLM_TOKEN_PARITY_LIMIT", "8"))
    return paths if limit == 0 else paths[:limit]


def _records(path: str) -> list[list[TraceRecord]]:
    with open(path, encoding="utf-8") as handle:
        episode = json.load(handle)
    session_id = str(episode.get("session_id") or episode.get("id") or os.path.basename(path))
    trajectories: list[list[TraceRecord]] = []
    for trajectory_index, trajectory in enumerate(episode.get("trajectories") or []):
        lineage_id = str(trajectory.get("uid") or f"trajectory-{trajectory_index}")
        records: list[TraceRecord] = []
        for index, step in enumerate(trajectory.get("steps") or []):
            chat = step.get("chat_completions") or []
            output = step.get("model_output") or {}
            prompt_ids = step.get("prompt_ids", output.get("prompt_ids"))
            completion_ids = step.get("response_ids", output.get("completion_ids"))
            if not chat or prompt_ids is None or completion_ids is None:
                continue
            records.append(
                TraceRecord(
                    trace_id=str(step.get("id") or f"{lineage_id}-{index}"),
                    session_id=session_id,
                    lineage_id=lineage_id,
                    messages=chat[:-1],
                    prompt_token_ids=prompt_ids,
                    response_message=chat[-1],
                    completion_token_ids=completion_ids,
                    logprobs=step.get("logprobs", output.get("logprobs")),
                    routing_matrices=step.get("routing_matrices", output.get("routing_matrices")),
                    finish_reason=output.get("finish_reason"),
                    weight_version=step.get("weight_version", output.get("weight_version")),
                    token_counts={"prompt": len(prompt_ids), "completion": len(completion_ids)},
                    timestamp=float(index),
                    metadata=step.get("metadata") or {},
                )
            )
        if records:
            trajectories.append(records)
    return trajectories


def _dump(record: TraceRecord) -> bytes:
    return json.dumps(record.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False).encode()


def _replay(records: list[TraceRecord], compact: bool) -> tuple[list[TraceRecord], dict | None]:
    store = MemoryTraceStore(compact=compact)
    for record in records:
        _run(store.store_trace(record.trace_id, record.session_id, record.model_dump(mode="python")))
    restored = [TraceRecord.model_validate(item) for item in _run(store.get_session_traces(records[0].session_id))]
    payload = _run(store.get_session_traces_compact(records[0].session_id)) if compact else None
    return restored, payload


def test_real_token_episode_store_and_wire_parity():
    paths = _paths()
    if not paths:
        pytest.skip("set RLLM_TOKEN_PARITY_EPISODES to real token-bearing episode JSON")

    trajectories = traces = prompt_tokens = completion_tokens = 0
    flat_bytes = compact_bytes = prompt_suffix_tokens = 0
    for path in paths:
        for records in _records(path):
            expected = [_dump(record) for record in records]
            default, _ = _replay(records, compact=False)
            compact, payload = _replay(records, compact=True)
            graph = TraceGraph.model_validate(json.loads(json.dumps(payload)))
            assert [_dump(record) for record in default] == expected
            assert [_dump(record) for record in compact] == expected
            assert [_dump(record) for record in graph.flatten()] == expected

            flat_payload = json.dumps([record.model_dump(mode="json") for record in records], separators=(",", ":"), ensure_ascii=False).encode()
            compact_payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
            assert len(compact_payload) < len(flat_payload)
            trajectories += 1
            traces += len(records)
            prompt_tokens += sum(len(record.prompt_token_ids) for record in records)
            completion_tokens += sum(len(record.completion_token_ids) for record in records)
            flat_bytes += len(flat_payload)
            compact_bytes += len(compact_payload)
            prompt_suffix_tokens += sum(len(delta.prompt_ids_suffix) for delta in graph.deltas)

    assert trajectories and traces and prompt_tokens and completion_tokens
    print(
        f"[token-parity] files={len(paths)} trajectories={trajectories} traces={traces} "
        f"prompt_tokens={prompt_tokens} prompt_suffix_tokens={prompt_suffix_tokens} "
        f"completion_tokens={completion_tokens} wire_ratio={flat_bytes / compact_bytes:.2f}x"
    )
