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
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../rllm/trainer/algorithms/config.py")
_spec = importlib.util.spec_from_file_location("rllm_common_config", _CONFIG_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AlgorithmConfig = _mod.AlgorithmConfig
rLLMAdvantageEstimator = _mod.rLLMAdvantageEstimator


def _make_config(norm_adv_by_std_in_grpo: bool = True, warmup_steps: int = -1):
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
                    "warmup_steps": warmup_steps,
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
    algo_config = AlgorithmConfig.from_config(config.rllm.algorithm, stepwise_advantage_mode=config.rllm.stepwise_advantage.mode)
    assert algo_config.norm_adv_by_std_in_grpo is True


def test_norm_adv_by_std_in_grpo_false_from_algorithm():
    """norm_adv_by_std_in_grpo=False is read from rllm.algorithm, not stepwise_advantage."""
    config = _make_config(norm_adv_by_std_in_grpo=False)
    algo_config = AlgorithmConfig.from_config(config.rllm.algorithm, stepwise_advantage_mode=config.rllm.stepwise_advantage.mode)
    assert algo_config.norm_adv_by_std_in_grpo is False


def test_warmup_steps_from_algorithm():
    config = _make_config(warmup_steps=25)
    algo_config = AlgorithmConfig.from_config(config.rllm.algorithm, stepwise_advantage_mode=config.rllm.stepwise_advantage.mode)
    assert algo_config.warmup_steps == 25


# --- ECHO (arXiv:2605.24517) -------------------------------------------------


def _echo_config(adv_estimator: str = "echo", env_loss_coef=None):
    section = {
        "adv_estimator": adv_estimator,
        "norm_adv_by_std_in_grpo": True,
    }
    if env_loss_coef is not None:
        section["env_loss_coef"] = env_loss_coef
    return OmegaConf.create({"rllm": {"algorithm": section, "stepwise_advantage": {"mode": "broadcast"}}})


def test_echo_estimator_resolves():
    """adv_estimator=echo resolves to the ECHO enum (not OTHER)."""
    config = _echo_config()
    algo_config = AlgorithmConfig.from_config(config.rllm.algorithm)
    assert algo_config.estimator == rLLMAdvantageEstimator.ECHO


def test_echo_defaults_lambda_to_paper_value():
    """echo with no explicit coef defaults env_loss_coef to the paper's 0.05."""
    algo_config = AlgorithmConfig.from_config(_echo_config().rllm.algorithm)
    assert algo_config.env_loss_coef == 0.05


def test_grpo_disables_env_loss_by_default():
    """Non-echo estimators leave env_loss_coef at 0.0 (plain GRPO, no env loss)."""
    algo_config = AlgorithmConfig.from_config(_echo_config(adv_estimator="grpo").rllm.algorithm)
    assert algo_config.env_loss_coef == 0.0


def test_echo_explicit_coef_overrides_default():
    """An explicit env_loss_coef wins over the echo default."""
    algo_config = AlgorithmConfig.from_config(_echo_config(env_loss_coef=0.02).rllm.algorithm)
    assert algo_config.env_loss_coef == 0.02


def test_env_loss_coef_can_enable_on_grpo():
    """env_loss_coef is the real switch: it can enable the env loss with adv_estimator=grpo."""
    algo_config = AlgorithmConfig.from_config(_echo_config(adv_estimator="grpo", env_loss_coef=0.05).rllm.algorithm)
    assert algo_config.env_loss_coef == 0.05


# --- DPPO ---------------------------------------------------------------------


def _dppo_config(**overrides):
    section = {
        "adv_estimator": "grpo",
        "norm_adv_by_std_in_grpo": True,
        "loss_fn": "dppo",
    }
    section.update(overrides)
    return OmegaConf.create({"rllm": {"algorithm": section, "stepwise_advantage": {"mode": "broadcast"}}})


def test_dppo_config_defaults():
    algo_config = AlgorithmConfig.from_config(_dppo_config().rllm.algorithm)
    assert algo_config.loss_fn == "dppo"
    assert algo_config.dppo_divergence_type == "tv"
    assert algo_config.dppo_divergence_threshold == 0.1


def test_dppo_config_overrides():
    algo_config = AlgorithmConfig.from_config(_dppo_config(dppo_divergence_type="kl", dppo_divergence_threshold=0.05).rllm.algorithm)
    assert algo_config.dppo_divergence_type == "kl"
    assert algo_config.dppo_divergence_threshold == 0.05


def test_dppo_rejects_invalid_divergence_type():
    with pytest.raises(ValueError, match="dppo_divergence_type"):
        AlgorithmConfig.from_config(_dppo_config(dppo_divergence_type="bad").rllm.algorithm)


def test_dppo_rejects_nonpositive_threshold():
    with pytest.raises(ValueError, match="dppo_divergence_threshold"):
        AlgorithmConfig.from_config(_dppo_config(dppo_divergence_threshold=0.0).rllm.algorithm)
