"""Episode codec: schema-2 compaction must be exactly lossless.

Unit tests cover the adversarial shapes observed in real dumps (repeated
identical content, non-cumulative rewrites, empty conversations); the parity
test then replays *real* saved episodes — discovered from
``~/.rllm/eval_results/*/episodes`` or ``RLLM_PARITY_DUMPS`` — and requires
``expand_episode(compact_episode(x))`` to be canonical-byte-identical to
``x``. It skips on machines without dumps (CI). ``RLLM_PARITY_LIMIT`` caps
episodes per dump (default 6; 0 = all).
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from rllm.eval.episode_codec import COMPACT_SCHEMA, SCHEMA_KEY, compact_episode, expand_episode


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()


def _episode(step_ccs, extra_step_fields=None):
    steps = [dict({"id": f"s{i}", "chat_completions": cc, "reward": 0.0}, **(extra_step_fields or {})) for i, cc in enumerate(step_ccs)]
    return {
        "id": "ep",
        "is_correct": True,
        "trajectories": [{"uid": "t", "name": "agent", "steps": steps, "reward": 1.0}],
        "metrics": {"f2p": 1.0},
    }


def _roundtrip(ep):
    compacted = compact_episode(ep)
    assert compacted[SCHEMA_KEY] == COMPACT_SCHEMA
    restored = expand_episode(compacted)
    assert _canonical(restored) == _canonical(ep)
    return compacted


def test_cumulative_roundtrip_and_dedup():
    convo = [{"role": "system", "content": "sys"}]
    ccs = []
    for i in range(5):
        convo = convo + [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
        ccs.append(list(convo))
    compacted = _roundtrip(_episode(ccs))
    assert len(compacted["trajectories"][0]["message_nodes"]) == len(convo)  # unique only


def test_repeated_identical_content_keeps_positions():
    ccs = [[{"role": "user", "content": "go"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "retry"}, {"role": "assistant", "content": ""}]]
    compacted = _roundtrip(_episode(ccs))
    assert len(compacted["trajectories"][0]["message_nodes"]) == 4


def test_noncumulative_rewrite_forks():
    ccs = [
        [{"role": "user", "content": "start"}, {"role": "assistant", "content": "v1"}],
        [{"role": "user", "content": "start"}, {"role": "assistant", "content": "SUMMARY"}, {"role": "user", "content": "next"}],
    ]
    compacted = _roundtrip(_episode(ccs))
    assert len(compacted["trajectories"][0]["message_nodes"]) == 4  # shared root + 3


def test_empty_and_missing_conversations():
    _roundtrip(_episode([[], [{"role": "user", "content": "hi"}]]))
    _roundtrip({"id": "ep", "trajectories": []})
    _roundtrip({"id": "ep"})


def test_compact_is_idempotent_and_expand_is_identity_on_legacy():
    ep = _episode([[{"role": "user", "content": "x"}]])
    once = compact_episode(ep)
    assert compact_episode(once) is once
    assert expand_episode(ep) is ep


def test_unknown_step_shape_kept_verbatim():
    ep = _episode([[{"role": "user", "content": "x"}]])
    ep["trajectories"].append({"uid": "weird", "steps": [{"chat_completions": "not-a-list"}]})
    compacted = compact_episode(ep)
    assert compacted["trajectories"][1] == ep["trajectories"][1]
    assert _canonical(expand_episode(compacted)) == _canonical(ep)


def _discover_dumps() -> list[str]:
    env = os.environ.get("RLLM_PARITY_DUMPS")
    if env:
        return [d for d in env.split(":") if os.path.isdir(d)]
    dirs = glob.glob(os.path.expanduser("~/.rllm/eval_results/*/episodes"))
    return sorted(d for d in dirs if glob.glob(os.path.join(d, "*.json")))


def test_real_dump_episode_parity():
    dumps = _discover_dumps()
    if not dumps:
        pytest.skip("no real eval dumps on this machine")
    limit = int(os.environ.get("RLLM_PARITY_LIMIT", "6"))

    checked = 0
    raw_bytes = compact_bytes = 0
    for dump in dumps:
        files = sorted(glob.glob(os.path.join(dump, "*.json")))
        if limit:
            stride = max(1, len(files) // limit)
            files = files[::stride][:limit]
        for path in files:
            with open(path, encoding="utf-8") as f:
                ep = json.load(f)
            # A dump may itself be schema-2 (written with RLLM_EPISODE_SCHEMA=2).
            # compact is identity on those, so round-trip through the EXPANDED
            # form — the invariant is expand(compact(x)) == x for schema-1 x.
            ep = expand_episode(ep)
            compacted = compact_episode(ep)
            restored = expand_episode(compacted)
            assert _canonical(restored) == _canonical(ep), f"not lossless: {path}"
            raw_bytes += os.path.getsize(path)
            compact_bytes += len(json.dumps(compacted))
            checked += 1
    assert checked > 0
    print(f"[parity-episode] {checked} real episodes lossless; {raw_bytes / 1e6:.0f} MB -> {compact_bytes / 1e6:.0f} MB ({raw_bytes / max(compact_bytes, 1):.0f}x)")
