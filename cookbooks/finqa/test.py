"""Tests for the finqa cookbook (no LLM, no real DB)."""

from __future__ import annotations

import os

# Skip the heavy SQLite preload at module import — tests don't touch real tables.
os.environ.setdefault("FINQA_SKIP_PRELOAD", "1")

from finqa_eval import _extract_final_answer, _table_access_score  # noqa: E402
from finqa_flow import _msg_to_dict, _truncate  # noqa: E402
from finqa_tools import TOOL_FNS, TOOL_SPECS, calculator  # noqa: E402

# -- Calculator ----------------------------------------------------------------


def test_calculator_basic():
    assert calculator("2 + 3") == 5.0
    assert calculator("(12 + 8) / 5") == 4.0


def test_calculator_power():
    # ^ should be normalized to ** (XOR vs power trap)
    assert calculator("2^3") == 8.0


def test_calculator_currency_and_commas():
    assert calculator("$1,234 + $5,678") == 6912.0


def test_calculator_percent():
    assert calculator("50% * 200") == 100.0


def test_calculator_invalid():
    out = calculator("not a number")
    assert isinstance(out, str)
    assert "Error" in out


# -- Tool spec / dispatch ------------------------------------------------------


def test_tool_specs_match_fns():
    spec_names = {s["function"]["name"] for s in TOOL_SPECS}
    assert spec_names == set(TOOL_FNS)


def test_tool_specs_have_required_fields():
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        fn = spec["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn


# -- FINAL ANSWER extraction ---------------------------------------------------


def test_extract_final_answer_code_block():
    text = "Reasoning here.\n\n```FINAL ANSWER: 42```"
    assert _extract_final_answer(text) == "42"


def test_extract_final_answer_paragraph():
    text = "Reasoning.\n\nFINAL ANSWER: 100\n\nMore notes follow."
    assert _extract_final_answer(text) == "100"


def test_extract_final_answer_tail():
    text = "Long reasoning. FINAL ANSWER: $1,234.56"
    assert _extract_final_answer(text) == "$1,234.56"


def test_extract_final_answer_missing():
    """Returns whole action when no FINAL ANSWER marker is present."""
    text = "I think the answer is 7."
    assert _extract_final_answer(text) == text


# -- Table-access scoring ------------------------------------------------------


def test_table_access_score_full_match():
    assert _table_access_score(["TableA", "TableB"], ["tablea", "tableb"]) == 1.0


def test_table_access_score_partial():
    assert _table_access_score(["TableA"], ["tablea", "tableb"]) == 0.5


def test_table_access_score_empty():
    assert _table_access_score([], ["tablea"]) == 0.0
    assert _table_access_score(["x"], []) == 0.0


# -- Flow helpers --------------------------------------------------------------


def test_truncate_short_string_unchanged():
    assert _truncate("short", n=100) == "short"


def test_truncate_long_string_clipped():
    s = "a" * 1000
    out = _truncate(s, n=200)
    assert "(truncated" in out
    assert len(out) < len(s)


def test_msg_to_dict_passthrough():
    """Already-dict messages should pass through unchanged."""
    d = {"role": "user", "content": "hi"}
    assert _msg_to_dict(d) == d
