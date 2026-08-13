"""Contract tests for the SFT ingestion bridges (``rllm.data.sft_bridges``).

Bridges turn raw dataset rows (plain OpenAI ``messages`` or ``<think>``-tagged
assistant turns) into schema ``SFTRow`` objects. Fixtures are built inline. RED
today: the module does not exist yet, so the import below fails at collection.
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

ROW_FLAGGED = {
    "messages": [
        {"role": "user", "content": "u1", "trainable": False},
        {"role": "assistant", "content": "<think>plan A</think>first", "trainable": True},
        {"role": "user", "content": "retry", "trainable": False},
        {"role": "assistant", "content": "<think>oops</think>bad", "trainable": False},
        {"role": "user", "content": "again", "trainable": False},
        {"role": "assistant", "content": "<think>plan C</think>done", "trainable": True},
    ]
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
    rows = bridge_think_tags([ROW_FLAGGED])
    assert len(rows) == 2
    targets = [next(message for message in row.messages if message.trainable) for row in rows]
    assert [target.thinking() for target in targets] == ["plan A", "plan C"]

    masked = [message for message in rows[-1].messages if message.text() == "bad"]
    assert len(masked) == 1
    assert masked[0].trainable is False
    assert masked[0].thinking() == ""


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
    assert [message.trainable for message in row.messages] == [False, True, False, False, False, True]


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
    assert [message.trainable for message in row.messages] == [False, True, False]


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
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "done",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "submit", "arguments": {"ok": True}}}],
        },
    ],
    "task_id": "t1",
}


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
def test_reasoning_sibling_field_becomes_thinking_part(key):
    message = {"role": "assistant", "content": "answer", key: "the chain of thought"}
    row = bridge_messages([{"messages": [{"role": "user", "content": "q"}, message]}])[0]
    assistant = row.messages[-1]
    assert [type(part) for part in assistant.content] == [ThinkingPart, TextPart]
    assert assistant.thinking() == "the chain of thought"
    assert assistant.text() == "answer"


def test_reasoning_only_turn_has_no_empty_text_part():
    assistant = bridge_messages([REASONING_ROW])[0].messages[-1]
    assert [type(part) for part in assistant.content] == [ThinkingPart]
    assert assistant.thinking() == "done"


@pytest.mark.parametrize(
    ("message_index", "call_id", "name", "arguments"),
    [(1, "c1", "bash", '{"cmd": "ls"}'), (-1, "c2", "submit", '{"ok": true}')],
)
def test_tool_calls_decoded_to_canonical_wire_shape(message_index, call_id, name, arguments):
    calls = bridge_messages([REASONING_ROW])[0].messages[message_index].tool_calls
    assert calls is not None and len(calls) == 1
    assert (calls[0].id, calls[0].function.name) == (call_id, name)
    assert calls[0].function.arguments == arguments


def test_unparseable_tool_calls_string_raises_naming_the_row():
    with pytest.raises(SFTSchemaError) as exc:
        bridge_messages([{"messages": [{"role": "assistant", "content": "x", "tool_calls": "not json"}]}])
    assert "row 0" in str(exc.value)
    assert "tool_calls is a string but not valid JSON" in str(exc.value)


def test_reasoning_on_prompt_turn_raises_instead_of_being_dropped():
    messages = [
        {"role": "user", "content": "q", "reasoning_content": "not mine"},
        {"role": "assistant", "content": "a"},
    ]
    with pytest.raises(SFTSchemaError, match="only on assistant"):
        bridge_messages([{"messages": messages}])


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_empty_reasoning_field_is_absent_on_any_role(key, role):
    message = bridge_messages([{"messages": [{"role": role, "content": "payload", key: ""}]}])[0].messages[0]
    assert message.text() == "payload"
    assert message.thinking() == ""


def test_conflicting_reasoning_aliases_raise_instead_of_dropping_one():
    message = {
        "role": "assistant",
        "content": "a",
        "reasoning_content": "first",
        "reasoning": "second",
    }
    with pytest.raises(SFTSchemaError, match="conflicting reasoning aliases"):
        bridge_messages([{"messages": [message]}])


@pytest.mark.parametrize("content", [7, {"text": "visible answer"}])
def test_reasoning_does_not_hide_unsupported_visible_content(content):
    message = {"role": "assistant", "content": content, "reasoning_content": "cot"}
    with pytest.raises(SFTSchemaError, match="assistant content must be"):
        bridge_messages([{"messages": [message]}])


def test_bridges_preserve_source_whitespace_verbatim():
    messages = [
        {"role": "system", "content": "sys prompt\n\n"},
        {"role": "user", "content": "  fix the bug\n"},
        {
            "role": "assistant",
            "content": "\n\nTHOUGHT: look around\n",
            "reasoning_content": "Let me explore.\n\n",
        },
    ]
    row = bridge_messages([{"messages": messages}])[0]
    assert row.messages[0].text() == "sys prompt\n\n"
    assert row.messages[1].text() == "  fix the bug\n"
    assert row.messages[2].thinking() == "Let me explore.\n\n"
    assert row.messages[2].text() == "\n\nTHOUGHT: look around\n"


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
