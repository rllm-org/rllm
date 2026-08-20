"""Tests for from-eval's reasoning-preserving automerge + CUSTOMIZED masking.

The builder is model-agnostic: reasoning is preserved as a structured
``ThinkingPart`` (the renderer picks the format), never a hardcoded ``<think>``.
"""

from __future__ import annotations

import json

from rllm.eval.curation import (
    CurationConfig,
    CurationStats,
    _episode_to_step_message_lists,
    _prefix_matches,
    _text_content,
    curate,
)
from rllm.eval.results import EvalItem, EvalResult


def _u(c):
    return {"role": "user", "content": c}


def _a(c, reasoning=""):
    m = {"role": "assistant", "content": c}
    if reasoning:
        m["reasoning_content"] = reasoning
    return m


def _ep(*step_ccs, name="terminus2"):
    return {"trajectories": [{"name": name, "steps": [{"chat_completions": cc} for cc in step_ccs]}]}


def _thinking(msg):
    return [p["thinking"] for p in msg["content"] if p.get("type") == "thinking"]


def test_thinking_trace_splits_per_turn():
    ep = _ep(
        [_u("task"), _a("act1", "plan1")],
        [_u("task"), _a("act1"), _u("obs1"), _a("act2", "plan2")],
        [_u("task"), _a("act1"), _u("obs1"), _a("act2"), _u("obs2"), _a("act3", "plan3")],
    )
    segs = _episode_to_step_message_lists(ep, None)
    assert len(segs) == 3  # per-turn: one row per step
    for seg in segs:
        assert sum(m["trainable"] for m in seg) == 1
        tgt = seg[-1]
        assert tgt["trainable"] and _thinking(tgt)  # reasoning preserved as ThinkingPart
        assert all(not m["trainable"] for m in seg[:-1])
    # reasoning is structured, NOT a hardcoded <think> string
    assert _thinking(segs[0][-1]) == ["plan1"]
    assert _text_content(segs[0][-1]["content"]) == "act1"


def test_non_thinking_trace_merges_into_one_row():
    ep = _ep(
        [_u("task"), _a("act1")],
        [_u("task"), _a("act1"), _u("obs1"), _a("act2")],
    )
    segs = _episode_to_step_message_lists(ep, None)
    assert len(segs) == 1  # no reasoning -> mergeable -> single row
    seg = segs[0]
    assert sum(m["trainable"] for m in seg) == 2
    assert [_text_content(m["content"]) for m in seg if m["trainable"]] == ["act1", "act2"]
    assert all(not _thinking(m) for m in seg if m["trainable"])


def test_every_step_trained_no_content_drops():
    ep = _ep(
        [_u("task"), _a("act1", "plan")],
        [_u("task"), _a("act1"), _u("o"), _a("act2")],  # no reasoning: still trained
    )
    segs = _episode_to_step_message_lists(ep, None)
    trained = [(_text_content(m["content"]), _thinking(m)) for s in segs for m in s if m["trainable"]]
    assert ("act2", []) in trained  # no-reasoning turn trained, no thinking
    assert ("act1", ["plan"]) in trained  # reasoning turn trained with structured thinking


def test_reasoning_turn_then_nonthinking_splits():
    # A reasoning turn seals its segment: the next turn can't merge (its thinking
    # would be stripped from history), so it starts a new row.
    ep = _ep(
        [_u("q"), _a("a1", "r1")],
        [_u("q"), _a("a1"), _u("o"), _a("a2")],
    )
    segs = _episode_to_step_message_lists(ep, None)
    assert len(segs) == 2
    assert _thinking(segs[0][-1]) == ["r1"] and _thinking(segs[1][-1]) == []


def test_context_reset_splits_no_loss():
    ep = _ep(
        [_u("q"), _a("a1")],
        [_u("q"), _a("a1"), _u("o1"), _a("a2")],
        [_u("SUMMARY of prior"), _a("a3")],  # history rewritten -> prefix breaks
    )
    segs = _episode_to_step_message_lists(ep, None)
    trained = [_text_content(m["content"]) for s in segs for m in s if m["trainable"]]
    assert trained == ["a1", "a2", "a3"]  # no turn lost


def test_interleaved_history_merges_and_keeps_thinking():
    # Interleaved-thinking run: history RETAINS each turn's reasoning, so the data
    # shows history == target form -> steps keep merging into one row (matching
    # inference, where past thinking stays in context).
    ep = _ep(
        [_u("task"), _a("act1", "plan1")],
        [_u("task"), _a("act1", "plan1"), _u("obs1"), _a("act2", "plan2")],
    )
    segs = _episode_to_step_message_lists(ep, None)
    assert len(segs) == 1
    seg = segs[0]
    assert [(_text_content(m["content"]), _thinking(m)) for m in seg if m["trainable"]] == [("act1", ["plan1"]), ("act2", ["plan2"])]


def test_interleaved_context_after_split_keeps_thinking():
    # A context reset still splits; the new row's interleaved history keeps its
    # ThinkingParts as untrained context — matching what the model actually saw.
    ep = _ep(
        [_u("A"), _a("a1", "r1")],
        [_u("RESET"), _a("a1", "r1"), _u("o"), _a("a2", "r2")],
    )
    segs = _episode_to_step_message_lists(ep, None)
    assert len(segs) == 2
    ctx_asst = [m for m in segs[1] if m["role"] == "assistant" and not m["trainable"]]
    assert [_thinking(m) for m in ctx_asst] == [["r1"]]


def test_tool_call_fields_preserved():
    # tool_calls / tool_call_id / name must survive the rebuild on both context
    # and target messages (native tool-calling harnesses).
    tc = [{"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]
    tc2 = [{"id": "call_2", "type": "function", "function": {"name": "stop", "arguments": "{}"}}]
    ep = _ep(
        [
            _u("q"),
            {"role": "assistant", "content": "", "tool_calls": tc},
            {"role": "tool", "tool_call_id": "call_1", "name": "run", "content": "out"},
            {"role": "assistant", "content": "done", "tool_calls": tc2},
        ]
    )
    (seg,) = _episode_to_step_message_lists(ep, None)
    asst_ctx = next(m for m in seg if m["role"] == "assistant" and not m["trainable"])
    tool_msg = next(m for m in seg if m["role"] == "tool")
    assert asst_ctx["tool_calls"] == tc
    assert tool_msg["tool_call_id"] == "call_1" and tool_msg["name"] == "run"
    assert seg[-1]["trainable"] and seg[-1]["tool_calls"] == tc2


def test_prefix_matches_primitive():
    seg = [
        {"role": "user", "content": [{"type": "text", "text": "task"}], "trainable": False},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}], "trainable": True},
    ]
    extends = [_u("task"), _a("a1"), _u("obs")]
    diverges = [_u("SUMMARY"), _a("a1")]
    assert _prefix_matches(seg, extends) is True
    assert _prefix_matches(seg, diverges) is False


# ---------------------------------------------------------------------------
# Regression tests for confirmed automerge-walk bugs (PR #739). Each MUST FAIL
# on current code for the documented reason and PASS once the planned fix lands.
# ---------------------------------------------------------------------------


def _write_eval_run(run_dir, *, attempts, rollouts):
    """Write a run dir in the on-disk eval format from full episode dicts.

    ``rollouts`` is a list of ``(task_idx, attempt, is_correct, episode_dict)``.
    Mirrors ``tests/eval/test_curation.py::_write_run`` but lets each rollout
    carry a full (possibly multi-step) episode instead of a single assistant
    string.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir()
    items = []
    for task_idx, attempt, is_correct, episode in rollouts:
        reward = 1.0 if is_correct else 0.0
        items.append(EvalItem(idx=task_idx, attempt=attempt, reward=reward, is_correct=is_correct, signals={"accuracy": reward}))
        eval_idx = task_idx * attempts + attempt
        (episodes_dir / f"episode_{eval_idx:06d}_t{task_idx}.json").write_text(json.dumps(episode))
    result = EvalResult.from_items("bench", "model", "agent", items, attempts=attempts)
    result.save(str(run_dir / "results.json"))
    return run_dir


def test_empty_history_message_does_not_break_merge():
    # F4: an empty (no-content, no-tool_calls) history message is dropped from the
    # running segment, which is then compared positionally against the RAW step
    # prefix -> the window desyncs and a clean prefix chain wrongly splits.
    # Non-thinking throughout => must stay ONE merged row training all 3 turns.
    empty_tool = {"role": "tool", "content": ""}
    ep = _ep(
        [_u("do task"), _a("run ls")],
        [_u("do task"), _a("run ls"), empty_tool, _a("run cat")],
        [_u("do task"), _a("run ls"), empty_tool, _a("run cat"), {"role": "tool", "content": "output"}, _a("done")],
    )
    segs = _episode_to_step_message_lists(ep, None)
    assert len(segs) == 1  # current code desyncs on the dropped empty and returns 2
    seg = segs[0]
    assert sum(m["trainable"] for m in seg) == 3
    assert [_text_content(m["content"]) for m in seg if m["trainable"]] == ["run ls", "run cat", "done"]


def test_dedup_does_not_collapse_distinct_tasks_with_identical_assistant(tmp_path):
    # F5: _assistant_signature hashes ONLY assistant content, so two different
    # tasks (different prompts) that happen to emit the same assistant text
    # collide under dedup and one task is silently dropped.
    rd = _write_eval_run(
        tmp_path / "cross",
        attempts=1,
        rollouts=[
            (0, 0, True, _ep([_u("list files in /app"), _a("ls")])),
            (1, 0, True, _ep([_u("show home dir"), _a("ls")])),
        ],
    )
    rows, _ = curate([rd], CurationConfig(dedup=True))
    assert len(rows) == 2  # current code dedups the two distinct tasks down to 1
    assert sorted(r["task_id"] for r in rows) == ["t0", "t1"]


def test_empty_final_assistant_not_trained():
    # F6a: an assistant turn with empty content and no tool_calls carries no
    # trainable signal and must not become a training target.
    ep = _ep([_u("hi"), _a("")])
    assert _episode_to_step_message_lists(ep, None) == []  # current code emits 1 empty-target segment


def test_curation_stats_track_merges_and_splits(tmp_path):
    # F6b: CurationStats must expose integer merge/split telemetry.
    stats = CurationStats()
    assert hasattr(stats, "segments_merged")
    assert hasattr(stats, "segments_split")
    assert stats.segments_merged == 0
    assert stats.segments_split == 0

    # A 3-step non-thinking episode is one clean prefix chain -> merges into a
    # single row, so at least one merge must be recorded.
    ep = _ep(
        [_u("task"), _a("act1")],
        [_u("task"), _a("act1"), _u("obs1"), _a("act2")],
        [_u("task"), _a("act1"), _u("obs1"), _a("act2"), _u("obs2"), _a("act3")],
    )
    rd = _write_eval_run(tmp_path / "merge", attempts=1, rollouts=[(0, 0, True, ep)])
    _, stats = curate([rd], CurationConfig())
    assert stats.segments_merged >= 1
