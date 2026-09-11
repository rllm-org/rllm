"""
Tests for AlgorithmConfig to verify norm_adv_by_std_in_grpo is read from
rllm.algorithm (not rllm.stepwise_advantage).

See: https://github.com/rllm-org/rllm/issues/447
"""

import importlib.util
import os

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


def _echo_config(adv_estimator: str = "echo", loss_fn=None):
    section = {
        "adv_estimator": adv_estimator,
        "norm_adv_by_std_in_grpo": True,
    }
    if loss_fn is not None:
        section["loss_fn"] = loss_fn
    return OmegaConf.create({"rllm": {"algorithm": section, "stepwise_advantage": {"mode": "broadcast"}}})


def test_echo_estimator_resolves():
    """adv_estimator=echo resolves to the ECHO enum (not OTHER)."""
    config = _echo_config()
    algo_config = AlgorithmConfig.from_config(config.rllm.algorithm)
    assert algo_config.estimator == rLLMAdvantageEstimator.ECHO


def test_echo_defaults_loss_fn_to_echo():
    """adv_estimator=echo defaults loss_fn to the `echo` loss (env_loss_coef now lives in loss_params)."""
    algo_config = AlgorithmConfig.from_config(_echo_config().rllm.algorithm)
    assert algo_config.loss_fn == "echo"


def test_grpo_leaves_loss_fn_unset():
    """Non-echo estimators get no default loss_fn (backend default / native kernel)."""
    algo_config = AlgorithmConfig.from_config(_echo_config(adv_estimator="grpo").rllm.algorithm)
    assert algo_config.loss_fn is None


def test_explicit_loss_fn_overrides_estimator_default():
    """An explicit loss_fn wins over the estimator's default (echo → dppo_tv here)."""
    algo_config = AlgorithmConfig.from_config(_echo_config(loss_fn="dppo_tv").rllm.algorithm)
    assert algo_config.loss_fn == "dppo_tv"
