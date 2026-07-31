"""Unit tests for the rllm eval / rllm train sampling-params resolution precedence."""

from __future__ import annotations

import pytest

from rllm.cli._sampling import resolve_eval_sampling, resolve_train_sampling

BASE = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 2048}


# -- eval: single config, empty base, always validation ---------------------


def test_eval_default_sampling():
    # Eval seeds temperature=1.0, top_p=1.0 (mirrors the cookbook training rollout).
    assert resolve_eval_sampling(None).as_dict() == {"temperature": 1.0, "top_p": 1.0}


def test_eval_string_only():
    # User params override the defaults (temperature) and add new keys (top_k);
    # the unspecified default (top_p) stays.
    assert resolve_eval_sampling("temperature=0.3,top_k=20").as_dict() == {"temperature": 0.3, "top_p": 1.0, "top_k": 20}


def test_eval_flags_beat_string():
    assert resolve_eval_sampling("temperature=0.3", temperature=0.9).temperature == 0.9


def test_eval_standalone_flags_only():
    assert resolve_eval_sampling(None, 0.5, 0.8, 256).as_dict() == {"temperature": 0.5, "top_p": 0.8, "max_tokens": 256}


def test_eval_flat_file(tmp_path):
    p = tmp_path / "sp.yaml"
    p.write_text("temperature: 0.42\npresence_penalty: 0.3\n")
    assert resolve_eval_sampling(f"@{p}").as_dict() == {"temperature": 0.42, "top_p": 1.0, "presence_penalty": 0.3}


def test_eval_nonmapping_file_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n")  # top-level list, not a mapping
    with pytest.raises(ValueError):
        resolve_eval_sampling(f"@{p}")


# -- train: (train, val) pair layered over base.yaml rollout.{train,val} -----


def test_train_base_only_passthrough():
    t, v = resolve_train_sampling(None, base_train=BASE, base_val=BASE)
    assert t.as_dict() == BASE and v.as_dict() == BASE


def test_train_string_overrides_base_for_both_incl_extra():
    t, v = resolve_train_sampling("temperature=0.6,presence_penalty=0.1", base_train=BASE, base_val=BASE)
    assert t.temperature == 0.6 and v.temperature == 0.6
    assert t.as_dict()["presence_penalty"] == 0.1 and v.as_dict()["presence_penalty"] == 0.1
    assert t.top_k == -1  # untouched base key kept


def test_train_flags_beat_string():
    t, v = resolve_train_sampling("temperature=0.6", temperature=0.2, base_train=BASE, base_val=BASE)
    assert t.temperature == 0.2 and v.temperature == 0.2
