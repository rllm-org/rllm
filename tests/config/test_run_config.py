"""Tests for the config-file loader (``rllm.config.run_config``)."""

from __future__ import annotations

import os

import pytest
from omegaconf import OmegaConf

from rllm.config.run_config import (
    RunSpec,
    _mirror_data,
    export_env,
    is_config_file,
    load_run_config,
    merge_backend_config,
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# is_config_file
# ---------------------------------------------------------------------------


def test_is_config_file(tmp_path):
    toml = _write(tmp_path, "run.toml", "backend = 'tinker'\n")
    assert is_config_file(str(toml))
    assert is_config_file(str(_write(tmp_path, "run.yaml", "backend: tinker\n")))
    # A bare benchmark name is not a file.
    assert not is_config_file("aime2024")
    # A directory is not a config file.
    assert not is_config_file(str(tmp_path))
    # A non-config suffix is not a config file.
    assert not is_config_file(str(_write(tmp_path, "notes.txt", "x")))


# ---------------------------------------------------------------------------
# RunSpec.from_raw
# ---------------------------------------------------------------------------


def test_runspec_nested_blocks():
    raw = {
        "backend": "fireworks",
        "run": {
            "agent": {"name": "terminus2", "args": {"max_turns": 75, "enable_summarize": False}},
            "dataset": {"train": "tb-opus-pass", "val": "terminal-bench@2.0", "val_split": "default", "max_examples": 10},
            "sandbox": {"backend": "modal", "concurrency": 8},
            "env": {"RLLM_HARNESS_RUN_TIMEOUT_S": "2400"},
        },
    }
    spec = RunSpec.from_raw(raw)
    assert spec.backend == "fireworks"
    assert spec.agent == "terminus2"
    assert spec.agent_args == {"max_turns": 75, "enable_summarize": False}
    assert spec.train_dataset == "tb-opus-pass"
    assert spec.val_dataset == "terminal-bench@2.0"
    assert spec.val_split == "default"
    assert spec.max_examples == 10
    assert spec.sandbox_backend == "modal"
    assert spec.sandbox_concurrency == 8
    assert spec.env == {"RLLM_HARNESS_RUN_TIMEOUT_S": "2400"}


def test_runspec_scalar_shorthands():
    # agent / sandbox may be given as bare strings.
    spec = RunSpec.from_raw({"run": {"agent": "react", "sandbox": "docker"}})
    assert spec.agent == "react"
    assert spec.agent_args == {}
    assert spec.sandbox_backend == "docker"
    assert spec.backend == "tinker"  # default


def test_runspec_evaluator_locations():
    # evaluator under [run.agent]
    a = RunSpec.from_raw({"run": {"agent": {"name": "react", "evaluator": "math"}}})
    assert a.evaluator == "math"
    # evaluator directly under [run]
    b = RunSpec.from_raw({"run": {"agent": "react", "evaluator": "math"}})
    assert b.evaluator == "math"


def test_runspec_entrypoint():
    spec = RunSpec.from_raw({"run": {"entrypoint": "my_pkg.setup:build"}})
    assert spec.entrypoint == "my_pkg.setup:build"


# ---------------------------------------------------------------------------
# merge_backend_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["tinker", "fireworks"])
def test_merge_backend_config_base(backend):
    cfg = merge_backend_config(backend)
    assert cfg.rllm.backend == backend
    assert cfg.rllm.algorithm.adv_estimator == "grpo"
    assert cfg.rllm.data.train_batch_size == 64


def test_merge_backend_config_user_override():
    cfg = merge_backend_config("tinker", {"rllm": {"trainer": {"project_name": "myproj"}}, "training": {"group_size": 16}})
    assert cfg.rllm.trainer.project_name == "myproj"
    # group_size flows into rllm.rollout.n via the template interpolation.
    assert OmegaConf.select(cfg, "rllm.rollout.n") == 16


def test_merge_backend_config_bad_backend():
    with pytest.raises(ValueError):
        merge_backend_config("nope")


# ---------------------------------------------------------------------------
# _mirror_data
# ---------------------------------------------------------------------------


def test_mirror_data_copies_into_rllm():
    tree = {"data": {"train_batch_size": 4, "max_prompt_length": 100}}
    out = _mirror_data(tree)
    assert out["rllm"]["data"]["train_batch_size"] == 4
    assert out["rllm"]["data"]["max_prompt_length"] == 100
    # original untouched
    assert out["data"]["train_batch_size"] == 4


def test_mirror_data_explicit_rllm_wins():
    tree = {"data": {"train_batch_size": 4}, "rllm": {"data": {"train_batch_size": 8}}}
    out = _mirror_data(tree)
    assert out["rllm"]["data"]["train_batch_size"] == 8  # explicit rllm.data.* wins


# ---------------------------------------------------------------------------
# load_run_config — end to end
# ---------------------------------------------------------------------------


def test_load_run_config_end_to_end(tmp_path):
    toml = _write(
        tmp_path,
        "run.toml",
        """
backend = "tinker"

[run.agent]
name = "react"
[run.agent.args]
max_turns = 5

[run.dataset]
train = "some-train"
val = "some-val"

[run.sandbox]
backend = "docker"

[model]
name = "Qwen/Qwen3-4B"

[data]
train_batch_size = 4

[rllm.trainer]
project_name = "cfg-test"
experiment_name = "exp1"
""",
    )
    cfg, run = load_run_config(str(toml))
    # RunSpec
    assert run.backend == "tinker"
    assert run.agent == "react"
    assert run.agent_args == {"max_turns": 5}
    assert run.train_dataset == "some-train"
    assert run.sandbox_backend == "docker"
    # Config tree
    assert cfg.model.name == "Qwen/Qwen3-4B"
    assert cfg.rllm.trainer.project_name == "cfg-test"
    assert cfg.rllm.trainer.experiment_name == "exp1"
    # [data] mirrored into [rllm.data]
    assert cfg.rllm.data.train_batch_size == 4
    assert cfg.data.train_batch_size == 4


def test_load_run_config_dotlist_overrides(tmp_path):
    toml = _write(tmp_path, "run.toml", 'backend = "tinker"\n[rllm.trainer]\nproject_name = "base"\n')
    cfg, _ = load_run_config(str(toml), overrides=["rllm.trainer.project_name=overridden", "training.group_size=32"])
    assert cfg.rllm.trainer.project_name == "overridden"
    assert OmegaConf.select(cfg, "rllm.rollout.n") == 32


def test_load_run_config_extends(tmp_path):
    _write(
        tmp_path,
        "base.toml",
        'backend = "tinker"\n[model]\nname = "base-model"\n[rllm.trainer]\nproject_name = "base-proj"\nsave_freq = 20\n',
    )
    child = _write(
        tmp_path,
        "child.toml",
        'extends = "base.toml"\n[model]\nname = "child-model"\n[rllm.trainer]\nsave_freq = 5\n',
    )
    cfg, run = load_run_config(str(child))
    assert run.backend == "tinker"  # inherited
    assert cfg.model.name == "child-model"  # child wins
    assert cfg.rllm.trainer.project_name == "base-proj"  # inherited
    assert cfg.rllm.trainer.save_freq == 5  # child wins


def test_load_run_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run_config(str(tmp_path / "nope.toml"))


def test_load_run_config_bad_backend(tmp_path):
    toml = _write(tmp_path, "run.toml", 'backend = "wat"\n')
    with pytest.raises(ValueError):
        load_run_config(str(toml))


# ---------------------------------------------------------------------------
# export_env
# ---------------------------------------------------------------------------


def test_export_env(monkeypatch):
    monkeypatch.delenv("RLLM_TEST_ENV_KEY", raising=False)
    export_env({"RLLM_TEST_ENV_KEY": 123})
    assert os.environ["RLLM_TEST_ENV_KEY"] == "123"  # stringified
