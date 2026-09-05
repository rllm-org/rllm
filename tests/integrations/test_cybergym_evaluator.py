"""CyberGymEvaluator: artifacts vs missing-trial infra error."""

from __future__ import annotations

import pytest

from rllm.integrations.cybergym.evaluator import CyberGymEvaluator
from rllm.types import Episode, Trajectory


def test_reads_cybergym_artifacts():
    episode = Episode(
        id="t",
        artifacts={
            "cybergym_trial_ran": True,
            "cybergym_reward": 1.0,
            "harbor_reward": 1.0,
            "vul_exit_code": 139,
            "fix_exit_code": 0,
            "n_submissions": 2,
            "cybergym_scoring": "any_of",
        },
    )
    out = CyberGymEvaluator().evaluate({"task_id": "arvo-libpng"}, episode)
    assert out.reward == 1.0
    assert out.is_correct is True
    assert out.metadata["vul_exit_code"] == 139
    assert out.metadata["n_submissions"] == 2


def test_reads_harbor_trial_artifacts():
    episode = Episode(id="t", artifacts={"harbor_trial_ran": True, "harbor_reward": 0.0})
    out = CyberGymEvaluator().evaluate({"task_id": "x"}, episode)
    assert out.reward == 0.0
    assert out.is_correct is False


def test_missing_trial_is_error_not_zero():
    episode = Episode(id="t", artifacts={})
    with pytest.raises(RuntimeError, match="not reward 0"):
        CyberGymEvaluator().evaluate({"task_id": "x"}, episode)


def test_trajectory_fallback():
    episode = Episode(
        id="t",
        artifacts={},
        trajectories=[Trajectory(name="harbor_trial", reward=1.0)],
    )
    out = CyberGymEvaluator().evaluate({"task_id": "x"}, episode)
    assert out.reward == 1.0
    assert out.metadata["eval_mode"] == "cybergym_trajectory_fallback"
