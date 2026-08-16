"""Real-data parity: compact store mode vs default mode vs the original bytes.

Replays the per-step conversation snapshots of real ``rllm eval`` episode
dumps through both store modes and requires exact byte parity:

    original -> store(compact) -> read  ==  original   (canonical JSON bytes)
    original -> store(default) -> read  ==  original   (canonical JSON bytes)

Real dumps — not mocks — are the fixture, because the adversarial shapes
(throttle-retry empty messages, non-cumulative rewrites, 200+ step sessions)
only occur in the wild. Dumps are discovered from ``RLLM_PARITY_DUMPS``
(colon-separated run dirs) or ``~/.rllm/eval_results/*/episodes``; the test
skips when none exist (CI has no dumps). ``RLLM_PARITY_LIMIT`` caps episodes
per dump (default 6 for test-suite runs; 0 = every episode, used for full
sweeps before a release).

Run a full sweep with:
    RLLM_PARITY_LIMIT=0 pytest tests/parity/test_store_real_dump_parity.py -s
"""

import asyncio
import glob
import json
import os

import pytest
from rllm_model_gateway.store.memory_store import MemoryTraceStore

_run = asyncio.run


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()


def _discover_dumps() -> list[str]:
    env = os.environ.get("RLLM_PARITY_DUMPS")
    if env:
        return [d for d in env.split(":") if os.path.isdir(d)]
    dirs = glob.glob(os.path.expanduser("~/.rllm/eval_results/*/episodes"))
    return sorted(d for d in dirs if glob.glob(os.path.join(d, "*.json")))


def _episode_records(path: str) -> list[dict]:
    """Trace inputs and outputs reconstructed from one legacy episode."""
    with open(path, encoding="utf-8") as f:
        ep = json.load(f)
    trajectories = ep.get("trajectories") or []
    if not trajectories:
        return []
    steps = trajectories[0].get("steps") or []
    seqs = [s.get("chat_completions") or [] for s in steps]
    # Legacy-form dumps only: compact-FORMAT episode files store marker dicts
    # in place of message lists, which are not per-call conversations (and the
    # schema-validating store rightly refuses them as messages).
    if not all(isinstance(s, list) and all(isinstance(m, dict) for m in s) for s in seqs):
        return []
    return [
        {
            "messages": chat[:-1],
            "prompt_token_ids": [],
            "response_message": chat[-1],
            "completion_token_ids": [],
        }
        for chat in seqs
        if chat
    ]


def _replay_and_check(records: list[dict], compact: bool) -> None:
    store = MemoryTraceStore(compact=compact)
    for i, record in enumerate(records):
        _run(store.store_trace(f"t{i}", "s", record))
    # Single-trace reads
    for i, record in enumerate(records):
        got = _run(store.get_trace(f"t{i}"))
        assert _canonical(got["messages"]) == _canonical(record["messages"])
        assert _canonical(got["response_message"]) == _canonical(record["response_message"])
    # Bulk read preserves order and content
    bulk = [(t["messages"], t["response_message"]) for t in _run(store.get_session_traces("s"))]
    expected = [(record["messages"], record["response_message"]) for record in records]
    assert _canonical(bulk) == _canonical(expected), f"bulk read (compact={compact}) not byte-identical"


def test_real_dump_parity_compact_vs_default():
    dumps = _discover_dumps()
    if not dumps:
        pytest.skip("no real eval dumps on this machine")
    limit = int(os.environ.get("RLLM_PARITY_LIMIT", "6"))

    episodes_checked = 0
    for dump in dumps:
        files = sorted(glob.glob(os.path.join(dump, "*.json")))
        if limit:
            # spread across the dump rather than taking the first N
            stride = max(1, len(files) // limit)
            files = files[::stride][:limit]
        for path in files:
            records = _episode_records(path)
            if not records:
                continue
            _replay_and_check(records, compact=True)
            _replay_and_check(records, compact=False)
            episodes_checked += 1
        print(f"[parity] {os.path.basename(os.path.dirname(dump))}: {len(files)} episodes checked")

    assert episodes_checked > 0, "dumps discovered but no episode had messages"
    print(f"[parity] total episodes byte-identical in both modes: {episodes_checked}")


def _replay_and_check_fetch(records: list[dict]) -> None:
    """Fetch-path parity: the wire TraceGraph flattens byte-identical to default."""
    from rllm_model_gateway.models import TraceGraph

    store = MemoryTraceStore(compact=True)
    for i, record in enumerate(records):
        _run(store.store_trace(f"t{i}", "s", record))

    default_form = [(t["messages"], t["response_message"]) for t in _run(store.get_session_traces("s"))]
    payload = _run(store.get_session_traces_compact("s"))
    # Simulate the wire: what the client sees is the JSON round-trip.
    payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    expanded = [(r.messages, r.response_message) for r in TraceGraph.model_validate(payload).flatten()]
    expected = [(record["messages"], record["response_message"]) for record in records]

    assert _canonical(expanded) == _canonical(expected)
    assert _canonical(expanded) == _canonical(default_form)


def test_real_dump_fetch_parity_compact_vs_default():
    dumps = _discover_dumps()
    if not dumps:
        pytest.skip("no real eval dumps on this machine")
    limit = int(os.environ.get("RLLM_PARITY_LIMIT", "6"))

    episodes_checked = 0
    for dump in dumps:
        files = sorted(glob.glob(os.path.join(dump, "*.json")))
        if limit:
            stride = max(1, len(files) // limit)
            files = files[::stride][:limit]
        for path in files:
            records = _episode_records(path)
            if not records:
                continue
            _replay_and_check_fetch(records)
            episodes_checked += 1
    assert episodes_checked > 0
    print(f"[parity-fetch] total episodes byte-identical over compact JSON wire: {episodes_checked}")
