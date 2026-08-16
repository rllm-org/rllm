"""Episode converter: compact-format compaction must be exactly lossless.

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

from rllm.eval.episode_compact import COMPACT_FORMAT, FORMAT_KEY, compact_episode, expand_episode


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
    assert compacted[FORMAT_KEY] == COMPACT_FORMAT
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
    if not (os.environ.get("RLLM_PARITY_DUMPS") or os.environ.get("RLLM_PARITY_SCAN") == "1"):
        pytest.skip("real-dump parity is opt-in: set RLLM_PARITY_SCAN=1 or RLLM_PARITY_DUMPS (scans local eval artifacts)")
    dumps = _discover_dumps()
    if not dumps:
        pytest.skip("no real eval dumps on this machine")
    max_bytes = int(os.environ.get("RLLM_PARITY_MAX_MB", "4000")) * 1_000_000
    scanned = 0
    limit = int(os.environ.get("RLLM_PARITY_LIMIT", "6"))

    checked = 0
    raw_bytes = compact_bytes = 0
    for dump in dumps:
        files = sorted(glob.glob(os.path.join(dump, "*.json")))
        if limit:
            stride = max(1, len(files) // limit)
            files = files[::stride][:limit]
        for path in files:
            scanned += os.path.getsize(path)
            if scanned > max_bytes:
                break
            with open(path, encoding="utf-8") as f:
                ep = json.load(f)
            # A dump may itself be compact-format (written with RLLM_EPISODE_FORMAT=compact).
            # compact is identity on those, so round-trip through the EXPANDED
            # form — the invariant is expand(compact(x)) == x for legacy-format x.
            raw = open(path, "rb").read()
            was_compact = ep.get("episode_format") == "compact"
            ep = expand_episode(ep)
            if was_compact:
                # A compact artifact must export TRUE legacy: no compact-only
                # keys may survive expansion (review: seven zero-step artifacts
                # kept message_nodes and one kept refs — invisible when the
                # expanded form is treated as its own baseline).
                s = json.dumps(ep)
                assert '"message_nodes"' not in s and '"messages_ref"' not in s and '"chat_completions_ref"' not in s, f"compact residue after expand: {path}"
            compacted = compact_episode(ep)
            restored = expand_episode(compacted)
            assert _canonical(restored) == _canonical(ep), f"not lossless: {path}"
            # Stricter: a legacy file converted and back must reproduce the
            # exact bytes the previous pipeline wrote (same writer settings) —
            # this catches key-order regressions canonical comparison hides.
            if not was_compact:
                import json as _json

                from rllm.eval.episode_store import _json_default

                back = _json.dumps(restored, indent=2, default=_json_default).encode()
                assert back == raw, f"not byte-identical: {path}"
            raw_bytes += os.path.getsize(path)
            compact_bytes += len(json.dumps(compacted))
            checked += 1
    assert checked > 0
    print(f"[parity-episode] {checked} real episodes lossless; {raw_bytes / 1e6:.0f} MB -> {compact_bytes / 1e6:.0f} MB ({raw_bytes / max(compact_bytes, 1):.0f}x)")


def test_reserved_keys_refuse_compaction_losslessly():
    ep = _episode([[{"role": "user", "content": "x"}]])
    ep["episode_format"] = "something-else"
    assert compact_episode(ep) is ep  # refused, unchanged
    ep2 = _episode([[{"role": "user", "content": "x"}]])
    ep2["trajectories"][0]["message_nodes"] = {"user": "data"}
    c = compact_episode(ep2)
    assert c["trajectories"][0] == ep2["trajectories"][0]  # trajectory verbatim
    assert _canonical(expand_episode(c)) == _canonical(ep2)


def test_expand_guards_cycles_and_dangling_refs():
    ep = _episode([[{"role": "user", "content": "x"}]])
    c = compact_episode(ep)
    nodes = c["trajectories"][0]["message_nodes"]
    nid = next(iter(nodes))
    broken = json.loads(json.dumps(c))
    broken["trajectories"][0]["message_nodes"][nid]["p"] = nid  # self-cycle
    with pytest.raises(ValueError, match="cycle"):
        expand_episode(broken)
    broken2 = json.loads(json.dumps(c))
    broken2["trajectories"][0]["message_nodes"][nid]["p"] = "missing-node"
    with pytest.raises(ValueError, match="dangling"):
        expand_episode(broken2)


def test_single_trigger_env_gates_episode_writes(tmp_path, monkeypatch):
    """RLLM_GATEWAY_STORE=compact is the ONLY trigger; unset writes byte-exact legacy."""
    import json as _json

    from rllm.eval.episode_store import EvalEpisodeStore, _json_default
    from rllm.types import Episode, Trajectory
    from rllm.types import Step as _Step

    ep = Episode(trajectories=[Trajectory(steps=[_Step(chat_completions=[{"role": "user", "content": "x"}])])])
    monkeypatch.delenv("RLLM_GATEWAY_STORE", raising=False)
    p1 = EvalEpisodeStore(tmp_path / "legacy").write(0, ep)
    legacy = p1.read_bytes()
    data = _json.loads(legacy)
    assert "episode_format" not in data
    # legacy output is exactly the pre-change writer's format (indent=2, same encoder)
    expected = _json.dumps(data, indent=2, default=_json_default).encode()
    assert legacy == expected

    monkeypatch.setenv("RLLM_GATEWAY_STORE", "compact")
    p2 = EvalEpisodeStore(tmp_path / "compact").write(0, ep)
    assert _json.loads(p2.read_bytes()).get("episode_format") == "compact"


def test_user_reserved_like_fields_survive_on_versioned_files():
    """Review P2c: healing is for unversioned pre-release artifacts ONLY."""
    # user's own compact_version at top level → compaction refuses, lossless
    ep = _episode([[{"role": "user", "content": "x"}]])
    ep["compact_version"] = "user-data"
    assert compact_episode(ep) is ep
    # versioned compact file: user-ish chat_completions_ref / empty message_nodes pass through
    base = compact_episode(_episode([[{"role": "user", "content": "x"}]]))
    base["trajectories"].append({"uid": "w", "steps": [{"chat_completions": [{"role": "u", "content": "y"}], "chat_completions_ref": ["custom"], "reward": 0.0}]})
    out = expand_episode(json.loads(json.dumps(base)))
    weird = out["trajectories"][1]["steps"][0]
    assert weird["chat_completions_ref"] == ["custom"]  # untouched
    assert weird["chat_completions"] == [{"role": "u", "content": "y"}]


def test_unknown_compact_version_rejected():
    ep = compact_episode(_episode([[{"role": "user", "content": "x"}]]))
    ep["compact_version"] = 999
    with pytest.raises(ValueError, match="unsupported"):
        expand_episode(ep)
