"""Tests for tinker checkpoint resume resolution (``_resume_step_dir``)."""

import json
import os

import pytest
from omegaconf import OmegaConf

from rllm.trainer.tinker.tinker_policy_trainer import TinkerPolicyTrainer


def _trainer(tmp_path, mode="auto", resume_from_path=None):
    # bypass __init__ (which needs a tinker service); only config-driven helpers are exercised
    t = object.__new__(TinkerPolicyTrainer)
    t.config = OmegaConf.create({"training": {"default_local_dir": str(tmp_path), "resume_mode": mode, "resume_from_path": resume_from_path}})
    return t


def _write_checkpoint(local_dir, step):
    step_dir = os.path.join(local_dir, f"global_step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    with open(os.path.join(step_dir, "checkpoint.json"), "w") as f:
        json.dump(
            {
                "name": f"{step:06d}",
                "state_path": f"tinker://u:train:0/weights/{step:06d}",
                "sampler_path": f"tinker://u:train:0/sampler_weights/{step:06d}",
                "dataloader_state": {"epoch": 0, "cursor": step, "seed": 0},
            },
            f,
        )
    with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(step))
    return step_dir


def test_auto_no_checkpoint_returns_none(tmp_path):
    assert _trainer(tmp_path, "auto")._resume_step_dir() is None


def test_auto_resumes_latest_from_tracker(tmp_path):
    _write_checkpoint(str(tmp_path), 20)
    step_dir = _write_checkpoint(str(tmp_path), 40)  # tracker now points at 40
    assert _trainer(tmp_path, "auto")._resume_step_dir() == step_dir


def test_auto_tracker_points_at_missing_dir_returns_none(tmp_path):
    with open(os.path.join(tmp_path, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("999")
    assert _trainer(tmp_path, "auto")._resume_step_dir() is None


def test_disable_returns_none_even_with_checkpoint(tmp_path):
    _write_checkpoint(str(tmp_path), 40)
    assert _trainer(tmp_path, "disable")._resume_step_dir() is None


def test_resume_path_explicit_folder(tmp_path):
    step_dir = _write_checkpoint(str(tmp_path), 40)
    assert _trainer(tmp_path, "resume_path", resume_from_path=step_dir)._resume_step_dir() == step_dir


def test_resume_path_missing_checkpoint_raises(tmp_path):
    # explicit resume must fail loudly, not silently fall back to scratch
    with pytest.raises(FileNotFoundError):
        _trainer(tmp_path, "resume_path", resume_from_path=str(tmp_path / "global_step_999"))._resume_step_dir()


def test_resume_path_requires_global_step_token(tmp_path):
    with pytest.raises(AssertionError):
        _trainer(tmp_path, "resume_path", resume_from_path=str(tmp_path / "bogus"))._resume_step_dir()
