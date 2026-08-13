"""Contract tests for the SFT ingestion bridges (``rllm.data.sft_bridges``).

Bridges turn raw dataset rows (plain OpenAI ``messages`` or ``<think>``-tagged
assistant turns) into schema ``SFTRow`` objects. Fixtures are built inline.
"""

from __future__ import annotations

import json

import pytest

from rllm.data.sft_bridges import BRIDGES, bridge_messages, bridge_think_tags, get_bridge
from rllm.data.sft_schema import SFTRow, SFTSchemaError, TextPart, ThinkingPart

# Two think-tagged rows: one multi-turn think-tagged conversation, plus a row whose
# assistant turn carries NO think tag.
ROW_THINK = {
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": '<think>\nplan A\n</think>\n\n{"cmd": "ls"}'},
        {"role": "user", "content": "out1"},
        {"role": "assistant", "content": "<think>\nplan B\n</think>\n\ndone"},
    ],
    "_task": "t1",
    "_group": "g",
    "_model": "opus",
    "_reward": 1,
}

ROW_NO_THINK = {
    "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "just an answer"},
    ],
    "_task": "t2",
    "_group": "g2",
    "_model": "opus",
    "_reward": 0,
}

# A pre-masked conversation: every message carries an explicit bool ``trainable``
# and the MIDDLE assistant turn is masked out (a parse-error/recovery step that
# must not be imitated). The two other assistant turns are trainable targets.
ROW_FLAGGED = {
    "messages": [
        {"role": "user", "content": "u1", "trainable": False},
        {"role": "assistant", "content": '<think>\nplan A\n</think>\n\n{"cmd": "ls"}', "trainable": True},
        {"role": "user", "content": "Previous response had parsing errors", "trainable": False},
        {"role": "assistant", "content": "<think>\noops\n</think>\n\nbadjson", "trainable": False},
        {"role": "user", "content": "out2", "trainable": False},
        {"role": "assistant", "content": "<think>\nplan C\n</think>\n\ndone", "trainable": True},
    ],
    "_task": "tf",
}


def _n_assistant(row):
    return sum(1 for m in row["messages"] if m["role"] == "assistant")


# --- think-tag extraction ----------------------------------------------------


def test_think_extraction_splits_cot_and_rest():
    row = bridge_think_tags([ROW_THINK], explode=False)[0]
    first = next(m for m in row.messages if m.role == "assistant")
    assert isinstance(first.content[0], ThinkingPart)
    assert first.content[0].thinking == "plan A"  # COT stripped
    assert first.thinking() == "plan A"
    assert isinstance(first.content[1], TextPart)
    assert first.text() == '{"cmd": "ls"}'  # REST left-stripped


def test_no_think_tag_assistant_is_text_only():
    row = bridge_think_tags([ROW_NO_THINK], explode=False)[0]
    asst = next(m for m in row.messages if m.role == "assistant")
    assert all(isinstance(p, TextPart) for p in asst.content)
    assert asst.thinking() == ""
    assert asst.text() == "just an answer"


# --- explode (default): one row per assistant turn ---------------------------


def test_explode_row_count_equals_assistant_turns():
    rows = bridge_think_tags([ROW_THINK])  # explode=True default
    assert len(rows) == _n_assistant(ROW_THINK) == 2


def test_explode_history_stripped_single_trainable_target():
    rows = bridge_think_tags([ROW_THINK])
    for row in rows:
        trainables = [m for m in row.messages if m.trainable]
        assert len(trainables) == 1  # exactly one target
        target = trainables[0]
        assert target.role == "assistant"
        assert row.messages[-1] is target  # target is the final turn
        assert any(isinstance(p, ThinkingPart) for p in target.content)  # target keeps its COT
        # History: all non-trainable, ThinkingParts removed.
        for m in row.messages[:-1]:
            assert m.trainable is False
            assert all(not isinstance(p, ThinkingPart) for p in m.content)

    # Targets track successive assistant turns; the 2nd row carries the 1st
    # assistant turn in history WITHOUT its ThinkingPart.
    targets = [next(m for m in r.messages if m.trainable) for r in rows]
    assert targets[0].thinking() == "plan A"
    assert targets[1].thinking() == "plan B"
    hist_asst = [m for m in rows[1].messages if m.role == "assistant" and not m.trainable]
    assert len(hist_asst) == 1
    assert all(not isinstance(p, ThinkingPart) for p in hist_asst[0].content)
    assert hist_asst[0].text() == '{"cmd": "ls"}'


def test_explode_honours_preexisting_trainable_flags():
    # A fully-flagged conversation explodes only its trainable assistant turns:
    # the masked middle step is never a target, but survives in later history
    # (thinking stripped, non-trainable) as a recovery example.
    rows = bridge_think_tags([ROW_FLAGGED])
    assert len(rows) == 2  # two trainable targets, NOT three assistant turns
    targets = [next(m for m in r.messages if m.trainable) for r in rows]
    assert [t.thinking() for t in targets] == ["plan A", "plan C"]
    for r in rows:
        assert sum(1 for m in r.messages if m.trainable) == 1

    # The masked assistant turn ("badjson") appears in the final row's history:
    # non-trainable and with its ThinkingPart removed — never emitted as a target.
    # (All prior assistant turns are non-trainable history here; the first
    # target's turn likewise appears CoT-stripped.)
    last = rows[-1]
    assert not any(m.trainable and m.text() == "badjson" for m in last.messages)
    badjson = [m for m in last.messages if m.role == "assistant" and m.text() == "badjson"]
    assert len(badjson) == 1
    assert badjson[0].trainable is False
    assert badjson[0].thinking() == ""
    assert all(not isinstance(p, ThinkingPart) for m in last.messages[:-1] for p in m.content)


# --- no-explode: one row per conversation ------------------------------------


def test_no_explode_single_row_all_assistant_trainable():
    rows = bridge_think_tags([ROW_THINK], explode=False)
    assert len(rows) == 1
    row = rows[0]
    for m in row.messages:
        if m.role == "assistant":
            assert m.trainable is True
            assert any(isinstance(p, ThinkingPart) for p in m.content)  # ThinkingParts KEPT
        else:
            assert m.trainable is False


def test_no_explode_honours_preexisting_trainable_flags():
    row = bridge_think_tags([ROW_FLAGGED], explode=False)[0]
    assert [m.trainable for m in row.messages] == [False, True, False, False, False, True]


def test_no_explode_partial_flags_trigger_full_mask_derivation():
    row = bridge_think_tags(
        [
            {
                "messages": [
                    {"role": "user", "content": "q", "trainable": False},
                    {"role": "assistant", "content": "<think>plan</think>answer", "trainable": False},
                    {"role": "user", "content": "follow-up"},
                ]
            }
        ],
        explode=False,
    )[0]
    assert [m.trainable for m in row.messages] == [False, True, False]


# --- row-level extras passthrough --------------------------------------------


def test_extras_passthrough_to_record():
    row = bridge_think_tags([ROW_THINK], explode=False)[0]
    rec = row.to_record()
    assert rec["_task"] == "t1"  # passed through verbatim
    assert rec["_reward"] == 1  # no rename
    assert rec["_group"] == "g"
    assert rec["_model"] == "opus"


def test_extras_carried_onto_exploded_rows():
    for r in bridge_think_tags([ROW_THINK]):
        assert r.to_record()["_task"] == "t1"


# --- bridge_messages: plain OpenAI rows --------------------------------------


PLAIN = [
    {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
    }
]


def test_bridge_messages_train_on_all():
    row = bridge_messages(PLAIN, train_on="all")[0]
    assert isinstance(row, SFTRow)
    assert [m.trainable for m in row.messages] == [False, True, False, True]
    # content coerced to a parts list.
    assert row.messages[1].text() == "a1"
    assert all(isinstance(p, TextPart) for p in row.messages[1].content)


def test_bridge_messages_train_on_last():
    row = bridge_messages(PLAIN, train_on="last")[0]
    assert [m.trainable for m in row.messages] == [False, False, False, True]


# --- source-shape normalization ----------------------------------------------

# The reasoning-model wire shape real exports produce: CoT in a sibling
# ``reasoning_content`` field, ``tool_calls`` serialized as a JSON string, and
# tool ``arguments`` stored already-parsed on the second call.
REASONING_ROW = {
    "messages": [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": "THOUGHT: look around",
            "reasoning_content": "Let me explore.",
            "tool_calls": json.dumps([{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]),
        },
        {"role": "tool", "content": "a.py"},
        {"role": "assistant", "content": "", "reasoning_content": "done", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "submit", "arguments": {"ok": True}}}]},
    ],
    "task_id": "t1",
}


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
def test_reasoning_sibling_field_becomes_thinking_part(key):
    """A sibling reasoning field is lifted ahead of the visible text instead of
    being dropped as an unknown message key."""
    msg = {"role": "assistant", "content": "answer", key: "the chain of thought"}
    row = bridge_messages([{"messages": [{"role": "user", "content": "q"}, msg]}])[0]
    asst = row.messages[-1]
    assert [type(p) for p in asst.content] == [ThinkingPart, TextPart]
    assert asst.thinking() == "the chain of thought"
    assert asst.text() == "answer"


def test_reasoning_only_turn_has_no_empty_text_part():
    """A pure tool-call turn (empty content) carrying reasoning renders as
    thinking alone — no empty text part is invented."""
    row = bridge_messages([REASONING_ROW])[0]
    last = row.messages[-1]
    assert [type(p) for p in last.content] == [ThinkingPart]
    assert last.thinking() == "done"


@pytest.mark.parametrize(
    ("msg_idx", "call_id", "name", "arguments"),
    [
        # The stringified list a parquet/HF writer produces, and a call whose
        # arguments were stored already-parsed.
        (1, "c1", "bash", '{"cmd": "ls"}'),
        (-1, "c2", "submit", '{"ok": true}'),
    ],
)
def test_tool_calls_decoded_to_canonical_wire_shape(msg_idx, call_id, name, arguments):
    calls = bridge_messages([REASONING_ROW])[0].messages[msg_idx].tool_calls
    assert calls is not None and len(calls) == 1
    assert (calls[0].id, calls[0].function.name) == (call_id, name)
    assert calls[0].function.arguments == arguments
    assert isinstance(json.loads(calls[0].function.arguments), dict)


def test_unparseable_tool_calls_string_raises_naming_the_row():
    with pytest.raises(SFTSchemaError) as exc:
        bridge_messages([{"messages": [{"role": "assistant", "content": "x", "tool_calls": "not json"}]}])
    assert "row 0" in str(exc.value)
    assert "tool_calls is a string but not valid JSON" in str(exc.value)


def test_reasoning_on_prompt_turn_raises_instead_of_being_dropped():
    messages = [{"role": "user", "content": "q", "reasoning_content": "not mine"}, {"role": "assistant", "content": "a"}]
    with pytest.raises(SFTSchemaError, match="only on assistant"):
        bridge_messages([{"messages": messages}])


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_empty_reasoning_field_is_absent_on_any_role(key, role):
    row = bridge_messages([{"messages": [{"role": role, "content": "payload", key: ""}]}])[0]
    message = row.messages[0]
    assert message.text() == "payload"
    assert message.thinking() == ""
    assert key not in row.to_record()["messages"][0]


def test_conflicting_reasoning_aliases_raise_instead_of_dropping_one():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "a",
            "reasoning_content": "first",
            "reasoning": "second",
        },
    ]
    with pytest.raises(SFTSchemaError, match="conflicting reasoning aliases"):
        bridge_messages([{"messages": messages}])


def test_non_string_reasoning_raises_rather_than_being_dropped():
    """Reasoning the bridge cannot lift is a hard error, never a silent drop."""
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": [{"type": "thinking", "thinking": "cot"}]}]
    with pytest.raises(SFTSchemaError) as exc:
        bridge_messages([{"messages": messages}])
    assert "reasoning_content must be a string" in str(exc.value)


@pytest.mark.parametrize("content", [7, {"text": "visible answer"}])
def test_reasoning_does_not_hide_unsupported_visible_content(content):
    """Lifting reasoning must not turn an invalid visible answer into a
    reasoning-only row by silently discarding it."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": content, "reasoning_content": "cot"},
    ]
    with pytest.raises(SFTSchemaError) as exc:
        bridge_messages([{"messages": messages}])
    assert "row 0" in str(exc.value)
    assert "assistant content must be a string, list, or null" in str(exc.value)


@pytest.mark.parametrize(
    ("bridge", "kwargs"),
    [(bridge_messages, {}), (bridge_think_tags, {"explode": False})],
)
def test_sibling_reasoning_conflicts_with_structural_inline_thinking(bridge, kwargs):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>\ninline\n</think>\n\nvisible", "reasoning_content": "sibling"},
    ]
    with pytest.raises(SFTSchemaError, match="sibling reasoning.*structural inline"):
        bridge([{"messages": messages}], **kwargs)


# --- structural inline thinking ---------------------------------------------


def test_plain_messages_rejects_structural_inline_thinking():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>inline</think>visible"},
    ]
    with pytest.raises(SFTSchemaError, match="think-tags.*thinking parts"):
        bridge_messages([{"messages": messages}])


def test_think_tags_remains_the_inline_thinking_parser():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>inline</think>visible"},
    ]
    asst = bridge_think_tags([{"messages": messages}], explode=False)[0].messages[-1]
    assert asst.thinking() == "inline"
    assert asst.text() == "visible"


def test_literal_think_tags_outside_structural_assistant_prefix_are_text():
    messages = [
        {"role": "user", "content": "<think>literal prompt text</think>"},
        {"role": "assistant", "content": "The literal is <think>not wire syntax</think>."},
    ]
    row = bridge_messages([{"messages": messages}])[0]
    assert [m.text() for m in row.messages] == [
        "<think>literal prompt text</think>",
        "The literal is <think>not wire syntax</think>.",
    ]


# --- source whitespace preservation -----------------------------------------

PADDED_MESSAGES = [
    {"role": "system", "content": "sys prompt\n\n"},
    {"role": "user", "content": "  fix the bug\n"},
    {"role": "assistant", "content": "\n\nTHOUGHT: look around\n", "reasoning_content": "Let me explore.\n\n"},
    {"role": "tool", "content": "a.py\tb.py\t\n"},
    {"role": "assistant", "content": "\n\ndone\n\n", "reasoning_content": "finished\n"},
]


def _bridge(name, messages, **kwargs):
    return get_bridge(name)([{"messages": messages}], **kwargs)


@pytest.mark.parametrize(("fmt", "opts"), [("messages", {}), ("think-tags", {"explode": False})])
def test_bridges_preserve_source_whitespace_verbatim(fmt, opts):
    row = _bridge(fmt, PADDED_MESSAGES, **opts)[0]
    assert row.messages[0].text() == "sys prompt\n\n"
    assert row.messages[1].text() == "  fix the bug\n"
    assert row.messages[2].thinking() == "Let me explore.\n\n"
    assert row.messages[2].text() == "\n\nTHOUGHT: look around\n"
    assert row.messages[3].text() == "a.py\tb.py\t\n"
    assert row.messages[4].thinking() == "finished\n"
    assert row.messages[4].text() == "\n\ndone\n\n"


# --- registry / get_bridge ---------------------------------------------------


def test_bridges_registry_keys():
    assert set(BRIDGES) == {"messages", "think-tags"}
    assert get_bridge("messages") is BRIDGES["messages"]
    assert get_bridge("think-tags") is BRIDGES["think-tags"]


def test_get_bridge_unknown_name_lists_valid():
    with pytest.raises(Exception) as exc:  # noqa: PT011 - message content is the contract
        get_bridge("bogus-format")
    msg = str(exc.value)
    assert "messages" in msg
    assert "think-tags" in msg
