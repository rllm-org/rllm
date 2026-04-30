"""Tests for ``propagate_rllm_to_verl_config`` in ``rllm.experimental.verl.utils``."""

import pytest
from omegaconf import OmegaConf


def _make_config():
    """Minimal config mirroring the post-compose unified config (verl backend).

    Only the fields the propagator touches are populated.
    """
    return OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "gae",  # verl /ppo_trainer default
                "norm_adv_by_std_in_grpo": False,
                "rollout_correction": {
                    "bypass_mode": False,
                    "rollout_is": None,
                    "rollout_is_threshold": 5.0,
                },
            },
            "actor_rollout_ref": {
                "actor": {
                    "kl_loss_coef": 0.0,
                    "loss_agg_mode": "token-mean",
                    "policy_loss": {"loss_mode": "vanilla"},
                    "clip_ratio": 0.2,
                    "clip_ratio_low": 0.2,
                    "clip_ratio_high": 0.2,
                    "use_kl_loss": False,
                },
                "rollout": {
                    "n": 1,
                    "val_kwargs": {"n": 1},
                },
            },
            "trainer": {
                "save_freq": -1,
                "test_freq": -1,
                "val_before_train": False,
                "val_only": False,
                "total_epochs": 1,
                "total_training_steps": -1,
                "logger": ["console"],
                "project_name": "verl",
                "experiment_name": "default",
            },
            "rllm": {
                "algorithm": {
                    "adv_estimator": "grpo",
                    "norm_adv_by_std_in_grpo": True,
                    "kl_beta": 0.0,
                    "loss_fn": None,
                    "loss_agg_mode": None,
                    "eps_clip": 0.2,
                    "eps_clip_high": None,
                    "rollout_correction": {
                        "bypass_mode": True,
                        "tis_mode": None,
                        "tis_cap": 5.0,
                    },
                },
                "rollout": {"n": 8, "n_val": 1},
                "trainer": {
                    "save_freq": 20,
                    "test_freq": 5,
                    "val_before_train": True,
                    "val_only": False,
                    "total_epochs": 10,
                    "total_batches": -1,
                    "logger": ["console"],
                    "project_name": "rllm-training",
                    "experiment_name": "default",
                },
            },
        }
    )


@pytest.fixture
def propagate(monkeypatch):
    """Returns a callable ``run(config, explicit_keys)`` that invokes the propagator
    with a stubbed ``_explicit_override_keys``."""
    from rllm.experimental.verl import utils as utils_mod

    def run(config, explicit_keys: set[str] | None = None):
        monkeypatch.setattr(utils_mod, "_explicit_override_keys", lambda *_a, **_k: set(explicit_keys or ()))
        utils_mod.propagate_rllm_to_verl_config(config)
        return config

    return run


# ---------------------------------------------------------------------------
# Single-variable, full precedence sweep on adv_estimator
# ---------------------------------------------------------------------------


def test_adv_estimator_default_uses_rllm_yaml(propagate):
    """No CLI overrides → both sides take rllm yaml default (grpo)."""
    cfg = propagate(_make_config())
    assert cfg.algorithm.adv_estimator == "grpo"
    assert cfg.rllm.algorithm.adv_estimator == "grpo"


def test_adv_estimator_rllm_cli_wins(propagate):
    """User typed rllm-side → propagates to verl."""
    cfg = _make_config()
    cfg.rllm.algorithm.adv_estimator = "rloo"  # simulate Hydra CLI override
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.adv_estimator"})
    assert cfg.algorithm.adv_estimator == "rloo"
    assert cfg.rllm.algorithm.adv_estimator == "rloo"


def test_adv_estimator_verl_cli_wins(propagate):
    """User typed verl-side → propagates to rllm."""
    cfg = _make_config()
    cfg.algorithm.adv_estimator = "rloo"
    cfg = propagate(cfg, explicit_keys={"algorithm.adv_estimator"})
    assert cfg.algorithm.adv_estimator == "rloo"
    assert cfg.rllm.algorithm.adv_estimator == "rloo"


def test_adv_estimator_rllm_wins_over_verl(propagate):
    """User typed both → rllm CLI takes precedence."""
    cfg = _make_config()
    cfg.rllm.algorithm.adv_estimator = "rloo"
    cfg.algorithm.adv_estimator = "reinforce"
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.adv_estimator", "algorithm.adv_estimator"})
    assert cfg.algorithm.adv_estimator == "rloo"
    assert cfg.rllm.algorithm.adv_estimator == "rloo"


def test_rllm_none_lets_verl_default_stand(propagate):
    """rllm value of None is treated as "no rllm default" — verl yaml stays."""
    cfg = _make_config()
    cfg.rllm.algorithm.loss_fn = None  # default
    cfg.actor_rollout_ref.actor.policy_loss.loss_mode = "vanilla"  # verl default
    cfg = propagate(cfg)
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"


# ---------------------------------------------------------------------------
# KL: kl_beta ↔ kl_loss_coef + derived use_kl_loss
# ---------------------------------------------------------------------------


def test_kl_beta_zero_disables_use_kl_loss(propagate):
    cfg = propagate(_make_config())
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == 0.0
    assert cfg.actor_rollout_ref.actor.use_kl_loss is False


def test_kl_beta_positive_enables_use_kl_loss(propagate):
    cfg = _make_config()
    cfg.rllm.algorithm.kl_beta = 0.01
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.kl_beta"})
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == 0.01
    assert cfg.actor_rollout_ref.actor.use_kl_loss is True


def test_explicit_use_kl_loss_overrides_derivation(propagate):
    """User explicitly sets actor.use_kl_loss=False → derivation skips even if kl_beta > 0."""
    cfg = _make_config()
    cfg.rllm.algorithm.kl_beta = 0.01
    cfg.actor_rollout_ref.actor.use_kl_loss = False
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.kl_beta", "actor_rollout_ref.actor.use_kl_loss"})
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == 0.01
    assert cfg.actor_rollout_ref.actor.use_kl_loss is False


def test_verl_kl_loss_coef_propagates_to_rllm_kl_beta(propagate):
    """User typed verl-side → kl_beta synced; use_kl_loss derived True."""
    cfg = _make_config()
    cfg.actor_rollout_ref.actor.kl_loss_coef = 0.02
    cfg = propagate(cfg, explicit_keys={"actor_rollout_ref.actor.kl_loss_coef"})
    assert cfg.rllm.algorithm.kl_beta == 0.02
    assert cfg.actor_rollout_ref.actor.use_kl_loss is True


# ---------------------------------------------------------------------------
# Clip ratio: clip_ratio / clip_ratio_low / clip_ratio_high precedence
# ---------------------------------------------------------------------------


def test_default_eps_clip_propagates_to_clip_ratio(propagate):
    """No overrides → verl.clip_ratio takes rllm yaml eps_clip."""
    cfg = propagate(_make_config())
    assert cfg.actor_rollout_ref.actor.clip_ratio == 0.2


def test_rllm_eps_clip_propagates_to_clip_ratio(propagate):
    """User sets rllm.eps_clip → verl.clip_ratio mirrors it (symmetric)."""
    cfg = _make_config()
    cfg.rllm.algorithm.eps_clip = 0.3
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.eps_clip"})
    assert cfg.actor_rollout_ref.actor.clip_ratio == 0.3


def test_verl_clip_ratio_propagates_to_rllm_eps_clip(propagate):
    """User sets verl.clip_ratio (no asymmetric override) → rllm.eps_clip mirrors it."""
    cfg = _make_config()
    cfg.actor_rollout_ref.actor.clip_ratio = 0.3
    cfg = propagate(cfg, explicit_keys={"actor_rollout_ref.actor.clip_ratio"})
    assert cfg.rllm.algorithm.eps_clip == 0.3


def test_clip_ratio_low_takes_precedence_over_clip_ratio(propagate):
    """User set both clip_ratio_low and clip_ratio → low wins as the effective bound."""
    cfg = _make_config()
    cfg.actor_rollout_ref.actor.clip_ratio_low = 0.15
    cfg.actor_rollout_ref.actor.clip_ratio = 0.3
    cfg = propagate(
        cfg,
        explicit_keys={
            "actor_rollout_ref.actor.clip_ratio_low",
            "actor_rollout_ref.actor.clip_ratio",
        },
    )
    assert cfg.rllm.algorithm.eps_clip == 0.15


def test_asymmetric_clip_propagates_to_rllm(propagate):
    """User typed clip_ratio_low/high explicitly → rllm.eps_clip / eps_clip_high reflect them."""
    cfg = _make_config()
    cfg.actor_rollout_ref.actor.clip_ratio_low = 0.2
    cfg.actor_rollout_ref.actor.clip_ratio_high = 0.28
    cfg = propagate(
        cfg,
        explicit_keys={
            "actor_rollout_ref.actor.clip_ratio_low",
            "actor_rollout_ref.actor.clip_ratio_high",
        },
    )
    assert cfg.rllm.algorithm.eps_clip == 0.2
    assert cfg.rllm.algorithm.eps_clip_high == 0.28
