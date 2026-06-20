"""Unit tests for rllm.cli._sampling.SamplingConfig."""

from __future__ import annotations

import pytest

from rllm.cli._sampling import SamplingConfig


def test_from_obj_splits_core_and_extra():
    cfg = SamplingConfig.from_obj({"temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.1, "min_p": 0.05})
    assert cfg.temperature == 0.6
    assert cfg.top_p == 0.95
    assert cfg.extra == {"presence_penalty": 0.1, "min_p": 0.05}


def test_from_obj_drops_none_values():
    cfg = SamplingConfig.from_obj({"temperature": 0.7, "top_p": None})
    assert cfg.as_dict() == {"temperature": 0.7}


def test_from_string_coerces_int_and_float():
    cfg = SamplingConfig.from_string("temperature=0.6,max_tokens=2048")
    assert cfg.temperature == 0.6 and isinstance(cfg.temperature, float)
    assert cfg.max_tokens == 2048 and isinstance(cfg.max_tokens, int)


def test_from_string_unknown_keys_go_to_extra():
    cfg = SamplingConfig.from_string("temperature=0.6,presence_penalty=0.1")
    assert cfg.temperature == 0.6
    assert cfg.extra == {"presence_penalty": 0.1}


def test_from_string_rejects_token_without_equals():
    with pytest.raises(ValueError, match="key=value"):
        SamplingConfig.from_string("temperature")


def test_merged_other_wins_and_keeps_untouched_keys():
    out = SamplingConfig.from_obj({"temperature": 1.0, "top_p": 1.0}).merged(SamplingConfig.from_obj({"temperature": 0.6}))
    assert out.temperature == 0.6  # override wins
    assert out.top_p == 1.0  # untouched kept


def test_merged_none_field_does_not_override():
    out = SamplingConfig.from_obj({"temperature": 1.0}).merged(SamplingConfig.from_obj({"top_p": 0.8}))
    assert out.temperature == 1.0 and out.top_p == 0.8


def test_merged_extra_shallow_merge():
    base = SamplingConfig.from_obj({"presence_penalty": 0.1, "min_p": 0.0})
    out = base.merged(SamplingConfig.from_obj({"min_p": 0.2, "frequency_penalty": 0.5}))
    assert out.extra == {"presence_penalty": 0.1, "min_p": 0.2, "frequency_penalty": 0.5}


def test_merged_precedence_chain():
    # base < file < string < flags
    out = (
        SamplingConfig.from_obj({"temperature": 1.0, "top_p": 1.0, "top_k": -1})
        .merged(SamplingConfig.from_obj({"temperature": 0.9}))
        .merged(SamplingConfig.from_string("temperature=0.7,presence_penalty=0.1"))
        .merged(SamplingConfig.from_obj({"temperature": 0.5}))
    )
    assert out.temperature == 0.5
    assert out.top_p == 1.0 and out.top_k == -1
    assert out.extra == {"presence_penalty": 0.1}


def test_as_dict_flattens_core_and_extra_unfiltered():
    # Every set key reaches the dict verbatim — no provider-specific filtering.
    cfg = SamplingConfig.from_obj({"temperature": 0.6, "top_k": 20, "presence_penalty": 0.1})
    assert cfg.as_dict() == {"temperature": 0.6, "top_k": 20, "presence_penalty": 0.1}
    assert SamplingConfig().is_empty
