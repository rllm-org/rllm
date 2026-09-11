"""Contract tests for the SFT schema (``rllm.data.sft_schema``).

Pure-pydantic — no tinker/verl. Covers message cleanup (str/None content
coercion, arrow-None artifact stripping), trainable-flag derivation policies,
validation errors, and the ``SFTRow`` record round-trip. RED today: the module
does not exist yet, so the import below fails at collection.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rllm.data.sft_schema import (
    SFTMessage,
    SFTRow,
    SFTSchemaError,
    SFTToolCall,
    TextPart,
    ThinkingPart,
    ToolFunction,
    normalize_message_dict,
    normalize_messages,
    normalize_row,
    normalize_rows,
)

# --- normalize_message_dict: pre-validation cleanup (returns a plain dict) ----


def test_normalize_message_dict_coerces_str_content():
    out = normalize_message_dict({"role": "user", "content": "hi"})
    assert out["content"] == [{"type": "text", "text": "hi"}]


def test_normalize_message_dict_coerces_none_content():
    out = normalize_message_dict({"role": "assistant", "content": None})
    assert out["content"] == [{"type": "text", "text": ""}]


def test_normalize_message_dict_drops_none_tool_fields():
    # Arrow round-trip fills absent fields with None; cleanup must strip them.
    out = normalize_message_dict({"role": "assistant", "content": "x", "tool_calls": None, "tool_call_id": None, "name": None})
    assert "tool_calls" not in out
    assert "tool_call_id" not in out
    assert "name" not in out


def test_normalize_message_dict_drops_none_crosskeys_in_parts():
    out = normalize_message_dict({"role": "user", "content": [{"type": "text", "text": "a", "thinking": None}]})
    assert out["content"] == [{"type": "text", "text": "a"}]


def test_normalize_message_dict_drops_unknown_keys():
    out = normalize_message_dict({"role": "user", "content": "x", "weight": 0.5, "bogus": "y"})
    assert "weight" not in out
    assert "bogus" not in out
    assert out["role"] == "user"


def test_normalize_message_dict_leaves_trainable_untouched():
    out = normalize_message_dict({"role": "assistant", "content": "x", "trainable": True})
    assert out["trainable"] is True


# --- normalize_messages: trainable derivation policies -----------------------


def _conv():
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_normalize_messages_derive_all():
    msgs = normalize_messages(_conv(), default_trainable="all")
    assert [m.trainable for m in msgs] == [False, False, True, False, True]


def test_normalize_messages_derive_last():
    msgs = normalize_messages(_conv(), default_trainable="last")
    # Only the LAST assistant turn is trainable under "last".
    assert [m.trainable for m in msgs] == [False, False, False, False, True]


def test_normalize_messages_already_flagged_untouched():
    # All messages carry real bools -> keep them, even a pattern derivation would
    # never produce (trainable user, non-trainable assistant). The requested
    # derivation policy is ignored for a complete mask.
    msgs = normalize_messages(
        [
            {"role": "user", "content": "u", "trainable": True},
            {"role": "assistant", "content": "a", "trainable": False},
        ],
        default_trainable="last",
    )
    assert [m.trainable for m in msgs] == [True, False]


@pytest.mark.parametrize("partial_value", [None, 1, "true"])
def test_normalize_messages_partial_flags_trigger_full_derive(partial_value):
    # ANY message lacking a bool flag -> derive the WHOLE conversation,
    # overriding even an explicit earlier target.
    msgs = normalize_messages(
        [
            {"role": "user", "content": "u", "trainable": partial_value},
            {"role": "assistant", "content": "a1", "trainable": True},
            {"role": "assistant", "content": "a2", "trainable": False},
        ],
        default_trainable="last",
    )
    assert [m.trainable for m in msgs] == [False, False, True]


def test_normalize_messages_content_is_parts_list():
    msgs = normalize_messages(_conv())
    assert all(isinstance(m, SFTMessage) for m in msgs)
    assert all(isinstance(p, TextPart) for p in msgs[0].content)
    assert msgs[2].text() == "a1"


def test_normalize_messages_rejects_unbridged_reasoning_field():
    with pytest.raises(SFTSchemaError, match="reasoning_content.*dataset import"):
        normalize_messages(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "important hidden reasoning"},
            ]
        )


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_normalize_messages_treats_empty_reasoning_as_absent(key, role):
    message = normalize_messages([{"role": role, "content": "payload", key: ""}])[0]
    assert message.text() == "payload"
    assert message.thinking() == ""


def test_canonical_text_part_may_contain_literal_think_tag():
    message = normalize_messages(
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>literal text</think>"}],
                "trainable": True,
            }
        ]
    )[0]
    assert message.text() == "<think>literal text</think>"


# --- validation errors (raised as SFTSchemaError by normalize_*) -------------


def test_unknown_part_type_rejected():
    with pytest.raises(SFTSchemaError):
        normalize_row({"messages": [{"role": "user", "content": [{"type": "bogus", "text": "x"}]}]})


def test_missing_role_rejected():
    with pytest.raises(SFTSchemaError):
        normalize_row({"messages": [{"content": "x"}]})


def test_empty_messages_rejected():
    with pytest.raises(SFTSchemaError):
        normalize_row({"messages": []})


def test_normalize_rows_error_mentions_row_index():
    rows = [
        {"messages": [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "fine"}]},
        {"messages": [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "fine"}]},
        {"messages": [{"content": "no role here"}]},  # bad row at index 2
    ]
    with pytest.raises(SFTSchemaError, match="2"):
        normalize_rows(rows)


# --- SFTMessage: strictness + helpers ----------------------------------------


def test_sftmessage_forbids_extra_keys_on_direct_construction():
    with pytest.raises(ValidationError):
        SFTMessage(role="user", content=[TextPart(text="x")], trainable=False, bogus=1)


def test_sftmessage_text_and_thinking_helpers():
    msg = SFTMessage(
        role="assistant",
        content=[ThinkingPart(thinking="plan"), TextPart(text="answer")],
        trainable=True,
    )
    assert msg.text() == "answer"
    assert msg.thinking() == "plan"


def test_thinking_part_type_literal():
    tp = ThinkingPart(thinking="cot")
    assert tp.type == "thinking"
    text = TextPart(text="hi")
    assert text.type == "text"


def test_tool_call_models_construct():
    tc = SFTToolCall(function=ToolFunction(name="run", arguments='{"cmd": "ls"}'))
    assert tc.type == "function"
    assert tc.id is None
    assert tc.function.name == "run"


# --- SFTRow.to_record round-trip ---------------------------------------------


def test_to_record_drops_none_fields_and_keeps_parts():
    row = normalize_row({"messages": [{"role": "assistant", "content": "hi"}]})
    rec = row.to_record()
    msg = rec["messages"][0]
    assert msg["content"] == [{"type": "text", "text": "hi"}]
    assert msg["trainable"] is True  # assistant, derived under default "all"
    # None fields are dropped by exclude_none.
    assert "tool_calls" not in msg
    assert "tool_call_id" not in msg
    assert "name" not in msg


def test_to_record_preserves_extra_row_fields():
    row = normalize_row(
        {
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "task_id": "t1",
            "reward": 1.0,
            "group": "g",
            "model": "opus",
        }
    )
    rec = row.to_record()
    assert rec["task_id"] == "t1"
    assert rec["reward"] == 1.0
    assert rec["group"] == "g"
    assert rec["model"] == "opus"


def test_normalize_row_returns_sftrow():
    row = normalize_row({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    assert isinstance(row, SFTRow)
    assert all(isinstance(m, SFTMessage) for m in row.messages)


def test_normalize_rows_returns_list_of_sftrow():
    rows = normalize_rows(
        [
            {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
            {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
        ]
    )
    assert len(rows) == 2
    assert all(isinstance(r, SFTRow) for r in rows)
