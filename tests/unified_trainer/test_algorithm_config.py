"""
Tests for AlgorithmConfig to verify norm_adv_by_std_in_grpo is read from
rllm.algorithm (not rllm.stepwise_advantage).

See: https://github.com/rllm-org/rllm/issues/447
"""

import importlib.util
import os

import pytest
from omegaconf import OmegaConf

# Import config module directly to avoid heavy transitive deps (codetiming, verl, etc.)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../rllm/experimental/common/config.py")
_spec = importlib.util.spec_from_file_location("rllm_common_config", _CONFIG_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AlgorithmConfig = _mod.AlgorithmConfig


def _make_config(norm_adv_by_std_in_grpo: bool = True):
    """Build a minimal DictConfig mirroring the real rllm config structure."""
    return OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo",
            },
            "rllm": {
                "algorithm": {
                    "adv_estimator": "grpo",
                    "norm_adv_by_std_in_grpo": norm_adv_by_std_in_grpo,
                    "use_precomputed_advantage": False,
                    "loss_fn": None,
                    "lr_schedule": "constant",
                    "warmup_steps_ratio": 0.0,
                },
                "stepwise_advantage": {
                    "mode": "broadcast",
                    # Intentionally omit norm_adv_by_std_in_grpo here to confirm
                    # the code reads from rllm.algorithm, not stepwise_advantage.
                },
            },
        }
    )


def test_norm_adv_by_std_in_grpo_true_from_algorithm():
    """norm_adv_by_std_in_grpo=True is read from rllm.algorithm, not stepwise_advantage."""
    config = _make_config(norm_adv_by_std_in_grpo=True)
    algo_config = AlgorithmConfig.from_config(config)
    assert algo_config.norm_adv_by_std_in_grpo is True


def test_norm_adv_by_std_in_grpo_false_from_algorithm():
    """norm_adv_by_std_in_grpo=False is read from rllm.algorithm, not stepwise_advantage."""
    config = _make_config(norm_adv_by_std_in_grpo=False)
    algo_config = AlgorithmConfig.from_config(config)
    assert algo_config.norm_adv_by_std_in_grpo is False
