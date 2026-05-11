"""Tests for ``rllm.parser.messages`` — TypedDicts, MessageList, MessageSnapshot,
and the ``to_openai`` / ``from_openai`` converters.

The TypedDicts themselves carry no runtime validation, so the "type" tests
just exercise the fact that vanilla dict literals satisfy them at runtime.
"""

from __future__ import annotations

import pytest

from rllm.parser.messages import (
    AssistantMessage,
    MessageList,
    Messages,
    SystemMessage,
    ToolMessage,
    UserMessage,
    from_openai,
    to_openai,
)

# ── G1.7.a — TypedDict construction (runtime is just dict) ───────────────


def test_typeddicts_are_dicts_at_runtime():
    sys_msg: SystemMessage = {"role": "system", "content": "you are helpful"}
    user_msg: UserMessage = {"role": "user", "content": "hi"}
    asst_msg: AssistantMessage = {
        "role": "assistant",
        "content": "hello!",
        "reasoning": "internal monologue",
    }
    asst_tool: AssistantMessage = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }
        ],
    }
    tool_msg: ToolMessage = {
        "role": "tool",
        "content": "3",
        "tool_call_id": "call_1",
        "name": "add",
    }
    for msg in (sys_msg, user_msg, asst_msg, asst_tool, tool_msg):
        assert isinstance(msg, dict)
        assert "role" in msg


# ── G1.7.b — plain list[dict] is accepted where Messages is expected ─────


def test_list_dict_satisfies_messages_alias():
    """A vanilla ``list[dict]`` is structurally a ``Messages`` at runtime."""
    raw: list[dict] = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    # Annotating as Messages is purely a typing hint, but at runtime the
    # values must be perfectly usable — exercise list/dict ops.
    typed: Messages = raw  # type: ignore[assignment]
    assert len(typed) == 3
    assert typed[0]["role"] == "system"
    assert [m["role"] for m in typed] == ["system", "user", "assistant"]


# ── G1.7.c / G1.7.d — MessageList append / snapshot semantics ────────────


def test_message_list_append_and_snapshot_length():
    ml = MessageList()
    ml.append({"role": "system", "content": "S"})
    ml.append({"role": "user", "content": "U1"})
    snap = ml.snapshot()
    assert len(snap) == 2

    # Appending after the snapshot must NOT change the snapshot length.
    ml.append({"role": "assistant", "content": "A1"})
    ml.append({"role": "user", "content": "U2"})
    assert len(snap) == 2
    assert len(ml) == 4

    # Iteration also stops at the snapshot length.
    roles = [m["role"] for m in snap]
    assert roles == ["system", "user"]


def test_message_list_extend():
    ml = MessageList([{"role": "system", "content": "S"}])
    ml.extend([{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}])
    assert len(ml) == 3
    assert ml[1]["role"] == "user"


def test_message_list_add_returns_plain_list():
    ml = MessageList([{"role": "user", "content": "U"}])
    combined = ml + [{"role": "assistant", "content": "A"}]
    assert isinstance(combined, list)
    assert len(combined) == 2
    # Adding shouldn't mutate the original.
    assert len(ml) == 1


def test_message_list_radd():
    ml = MessageList([{"role": "user", "content": "U"}])
    combined = [{"role": "system", "content": "S"}] + ml
    assert isinstance(combined, list)
    assert [m["role"] for m in combined] == ["system", "user"]


def test_snapshot_index_and_slice():
    ml = MessageList(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
    )
    snap = ml.snapshot()
    assert snap[0]["role"] == "system"
    assert snap[-1]["role"] == "assistant"
    sliced = snap[:2]
    assert [m["role"] for m in sliced] == ["system", "user"]
    with pytest.raises(IndexError):
        _ = snap[5]


# ── G1.7.e — snapshot.to_list() shares dicts (shallow contract) ──────────


def test_snapshot_to_list_shares_dicts():
    msg = {"role": "user", "content": "U"}
    ml = MessageList([msg])
    snap = ml.snapshot()
    restored = snap.to_list()
    assert restored == [msg]
    # Shallow: the dict in `restored` is the *same object* as the original.
    assert restored[0] is msg


def test_snapshot_to_list_independence_from_future_appends():
    ml = MessageList([{"role": "user", "content": "U"}])
    snap = ml.snapshot()
    captured = snap.to_list()
    ml.append({"role": "assistant", "content": "A"})
    assert len(captured) == 1
    assert len(snap) == 1
    assert len(ml) == 2


# ── G1.7.f — to_openai / from_openai roundtrip ───────────────────────────


def test_to_openai_drops_reasoning():
    msgs: Messages = [
        {"role": "assistant", "content": "answer", "reasoning": "internal"},
    ]
    out = to_openai(msgs)
    assert out == [{"role": "assistant", "content": "answer"}]


def test_to_openai_preserves_tool_calls():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1}'},
        }
    ]
    msgs: Messages = [
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
    ]
    out = to_openai(msgs)
    assert out[0]["tool_calls"] == tool_calls


def test_to_openai_accepts_message_list_and_snapshot():
    ml = MessageList(
        [
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A", "reasoning": "drop me"},
        ]
    )
    out_list = to_openai(ml)
    out_snap = to_openai(ml.snapshot())
    assert out_list == out_snap
    assert "reasoning" not in out_list[1]


def test_from_openai_validates_structure():
    with pytest.raises(TypeError):
        from_openai(["not a dict"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        from_openai([{"content": "no role"}])


def test_roundtrip_to_openai_from_openai():
    msgs: Messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A", "reasoning": "internal"},
        {"role": "tool", "content": "T", "tool_call_id": "x", "name": "foo"},
    ]
    out = to_openai(msgs)
    back = from_openai(out)
    # `reasoning` is dropped on the way out — that's expected.
    assert back == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "tool", "content": "T", "tool_call_id": "x", "name": "foo"},
    ]


# ── G1.7.g — from_openai normalizes reasoning_content → reasoning ────────


def test_from_openai_normalizes_reasoning_content():
    incoming = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "vllm-style thinking",
        }
    ]
    out = from_openai(incoming)
    assert "reasoning_content" not in out[0]
    assert out[0].get("reasoning") == "vllm-style thinking"


def test_from_openai_drops_alias_when_canonical_present():
    incoming = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "canonical",
            "reasoning_content": "alias",
        }
    ]
    out = from_openai(incoming)
    assert out[0].get("reasoning") == "canonical"
    assert "reasoning_content" not in out[0]


def test_from_openai_preserves_unknown_fields():
    incoming = [{"role": "user", "content": "U", "custom_metadata": {"x": 1}}]
    out = from_openai(incoming)
    assert out[0].get("custom_metadata") == {"x": 1}


# ── Bonus: shallow-share contract is observable (documented behavior) ────


def test_shallow_share_visible_through_snapshot():
    """Documenting the shallow contract: in-place edits leak across snapshots.

    This test exists to prevent silent regressions: if anyone "fixes" the
    snapshot to deep-copy, this assertion will flip to passing inversely
    and force a docstring / contract update.
    """
    msg = {"role": "user", "content": "U"}
    ml = MessageList([msg])
    snap = ml.snapshot()
    msg["content"] = "MUTATED"
    assert snap[0]["content"] == "MUTATED"
