"""Tests for from-eval's reasoning-preserving automerge + CUSTOMIZED masking.

The builder is model-agnostic: reasoning is preserved as a structured
``ThinkingPart`` (the renderer picks the format), never a hardcoded ``<think>``.
"""

from __future__ import annotations

from rllm.eval.curation import _episode_to_step_message_lists, _prefix_matches, _text_content
from rllm.trainer.sft.tinker_dataset import _ensure_trainable


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


def test_prefix_matches_primitive():
    seg = [
        {"role": "user", "content": [{"type": "text", "text": "task"}], "trainable": False},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}], "trainable": True},
    ]
    extends = [_u("task"), _a("a1"), _u("obs")]
    diverges = [_u("SUMMARY"), _a("a1")]
    assert _prefix_matches(seg, extends) is True
    assert _prefix_matches(seg, diverges) is False


def test_ensure_trainable_passthrough_and_derive():
    flagged = [{"role": "user", "content": "x", "trainable": False}, {"role": "assistant", "content": "a", "trainable": True}]
    assert _ensure_trainable(flagged, last_only=False) is flagged
    plain = [_u("q"), _a("a1"), _u("o"), _a("a2")]
    assert [m["trainable"] for m in _ensure_trainable(plain, last_only=False)] == [False, True, False, True]
    assert [m["trainable"] for m in _ensure_trainable(plain, last_only=True)] == [False, False, False, True]
