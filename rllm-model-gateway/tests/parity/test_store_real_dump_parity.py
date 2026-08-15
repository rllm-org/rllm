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


def _episode_step_messages(path: str) -> list[list[dict]]:
    """Per-step conversation snapshots of one episode (may be non-cumulative)."""
    with open(path, encoding="utf-8") as f:
        ep = json.load(f)
    trajectories = ep.get("trajectories") or []
    if not trajectories:
        return []
    steps = trajectories[0].get("steps") or []
    return [s.get("chat_completions") or [] for s in steps]


def _replay_and_check(seqs: list[list[dict]], compact: bool) -> None:
    store = MemoryTraceStore(compact=compact)
    for i, cc in enumerate(seqs):
        _run(store.store_trace(f"t{i}", "s", {"messages": cc, "response_message": {}}))
    # Single-trace reads
    for i, cc in enumerate(seqs):
        got = _run(store.get_trace(f"t{i}"))["messages"]
        assert _canonical(got) == _canonical(cc), f"trace {i} (compact={compact}) not byte-identical"
    # Bulk read preserves order and content
    bulk = [t["messages"] for t in _run(store.get_session_traces("s"))]
    assert _canonical(bulk) == _canonical(seqs), f"bulk read (compact={compact}) not byte-identical"


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
            seqs = _episode_step_messages(path)
            if not any(seqs):
                continue
            _replay_and_check(seqs, compact=True)
            _replay_and_check(seqs, compact=False)
            episodes_checked += 1
        print(f"[parity] {os.path.basename(os.path.dirname(dump))}: {len(files)} episodes checked")

    assert episodes_checked > 0, "dumps discovered but no episode had messages"
    print(f"[parity] total episodes byte-identical in both modes: {episodes_checked}")


def _replay_and_check_fetch(seqs: list[list[dict]], compact_store: bool) -> None:
    """Fetch-path parity: compact wire payload expands byte-identical to default."""
    from rllm_model_gateway.client import _expand_compact_traces

    store = MemoryTraceStore(compact=compact_store)
    for i, cc in enumerate(seqs):
        _run(store.store_trace(f"t{i}", "s", {"messages": cc, "response_message": {}}))

    default_form = [t["messages"] for t in _run(store.get_session_traces("s"))]
    payload = _run(store.get_session_traces_compact("s"))
    # Simulate the wire: what the client sees is the JSON round-trip.
    payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    expanded = [t["messages"] for t in _expand_compact_traces(payload)]

    assert _canonical(expanded) == _canonical(seqs), f"compact fetch (store compact={compact_store}) != original"
    assert _canonical(expanded) == _canonical(default_form), f"compact fetch (store compact={compact_store}) != default fetch"


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
            seqs = _episode_step_messages(path)
            if not any(seqs):
                continue
            _replay_and_check_fetch(seqs, compact_store=True)
            _replay_and_check_fetch(seqs, compact_store=False)
            episodes_checked += 1
    assert episodes_checked > 0
    print(f"[parity-fetch] total episodes byte-identical (both store modes, over JSON wire): {episodes_checked}")
