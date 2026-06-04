"""Tests for ``sync_config`` in ``rllm.experimental.verl.utils``."""

import logging

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
                    "optim": {
                        "lr_scheduler_type": "constant",
                        "lr_warmup_steps": -1,
                        "lr_warmup_steps_ratio": 0.0,
                    },
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
                    "lr_schedule": "constant",
                    "warmup_steps": -1,
                    "warmup_steps_ratio": 0.0,
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
        utils_mod.sync_config(config)
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


def test_adv_estimator_verl_cli_wins_with_deprecation_warning(propagate, caplog):
    """User typed verl-side → propagates to rllm for now, with a warning."""
    caplog.set_level(logging.WARNING, logger="rllm.experimental.verl.utils")
    cfg = _make_config()
    cfg.algorithm.adv_estimator = "rloo"
    cfg = propagate(cfg, explicit_keys={"algorithm.adv_estimator"})
    assert cfg.algorithm.adv_estimator == "rloo"
    assert cfg.rllm.algorithm.adv_estimator == "rloo"
    assert "algorithm.adv_estimator is deprecated" in caplog.text
    assert "rllm.algorithm.adv_estimator" in caplog.text


def test_adv_estimator_rllm_wins_over_verl_conflict(propagate, caplog):
    """User typed both → rllm CLI takes precedence."""
    caplog.set_level(logging.WARNING, logger="rllm.experimental.verl.utils")
    cfg = _make_config()
    cfg.rllm.algorithm.adv_estimator = "rloo"
    cfg.algorithm.adv_estimator = "reinforce"
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.adv_estimator", "algorithm.adv_estimator"})
    assert cfg.algorithm.adv_estimator == "rloo"
    assert cfg.rllm.algorithm.adv_estimator == "rloo"
    assert "algorithm.adv_estimator conflicts with rllm.algorithm.adv_estimator" in caplog.text


def test_rllm_none_lets_verl_default_stand(propagate):
    """rllm value of None is treated as "no rllm default" — verl yaml stays."""
    cfg = _make_config()
    cfg.rllm.algorithm.loss_fn = None  # default
    cfg.actor_rollout_ref.actor.policy_loss.loss_mode = "vanilla"  # verl default
    cfg = propagate(cfg)
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"


def test_total_batches_maps_to_verl_total_training_steps(propagate):
    cfg = _make_config()
    cfg.rllm.trainer.total_batches = 42
    cfg = propagate(cfg, explicit_keys={"rllm.trainer.total_batches"})
    assert cfg.trainer.total_training_steps == 42

    cfg = _make_config()
    cfg.rllm.trainer.total_batches = -1
    cfg = propagate(cfg)
    assert cfg.trainer.total_training_steps is None

    cfg = _make_config()
    cfg.rllm.trainer.total_batches = 0
    cfg = propagate(cfg, explicit_keys={"rllm.trainer.total_batches"})
    assert cfg.trainer.total_training_steps is None


def test_verl_total_training_steps_syncs_to_rllm_total_batches_with_warning(propagate, caplog):
    caplog.set_level(logging.WARNING, logger="rllm.experimental.verl.utils")
    cfg = _make_config()
    cfg.trainer.total_training_steps = 42
    cfg = propagate(cfg, explicit_keys={"trainer.total_training_steps"})
    assert cfg.trainer.total_training_steps == 42
    assert cfg.rllm.trainer.total_batches == 42
    assert "trainer.total_training_steps is deprecated" in caplog.text


def test_sync_before_resolve_updates_interpolated_verl_paths(propagate):
    cfg = _make_config()
    cfg.trainer.default_local_dir = "checkpoints/${trainer.project_name}/${trainer.experiment_name}"
    cfg.rllm.trainer.project_name = "project"
    cfg.rllm.trainer.experiment_name = "experiment"
    cfg = propagate(cfg, explicit_keys={"rllm.trainer.project_name", "rllm.trainer.experiment_name"})
    OmegaConf.resolve(cfg)
    assert cfg.trainer.default_local_dir == "checkpoints/project/experiment"


# ---------------------------------------------------------------------------
# LR schedule / warmup: rllm.algorithm ↔ verl actor optimizer
# ---------------------------------------------------------------------------


def test_rllm_lr_schedule_propagates_to_verl_optimizer(propagate):
    cfg = _make_config()
    cfg.rllm.algorithm.lr_schedule = "cosine"
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.lr_schedule"})
    assert cfg.actor_rollout_ref.actor.optim.lr_scheduler_type == "cosine"


def test_verl_lr_scheduler_type_propagates_to_rllm_with_warning(propagate, caplog):
    caplog.set_level(logging.WARNING, logger="rllm.experimental.verl.utils")
    cfg = _make_config()
    cfg.actor_rollout_ref.actor.optim.lr_scheduler_type = "cosine"
    cfg = propagate(cfg, explicit_keys={"actor_rollout_ref.actor.optim.lr_scheduler_type"})
    assert cfg.rllm.algorithm.lr_schedule == "cosine"
    assert "actor_rollout_ref.actor.optim.lr_scheduler_type is deprecated" in caplog.text


def test_lr_schedule_sync_does_not_validate_backend_values(propagate):
    cfg = _make_config()
    cfg.rllm.algorithm.lr_schedule = "linear"
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.lr_schedule"})
    assert cfg.actor_rollout_ref.actor.optim.lr_scheduler_type == "linear"


def test_rllm_warmup_steps_propagates_to_verl_optimizer(propagate):
    cfg = _make_config()
    cfg.rllm.algorithm.warmup_steps = 25
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.warmup_steps"})
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 25


def test_verl_warmup_steps_propagates_to_rllm_with_warning(propagate, caplog):
    caplog.set_level(logging.WARNING, logger="rllm.experimental.verl.utils")
    cfg = _make_config()
    cfg.actor_rollout_ref.actor.optim.lr_warmup_steps = 25
    cfg = propagate(cfg, explicit_keys={"actor_rollout_ref.actor.optim.lr_warmup_steps"})
    assert cfg.rllm.algorithm.warmup_steps == 25
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps is deprecated" in caplog.text


def test_rllm_warmup_steps_ratio_propagates_to_verl_optimizer(propagate):
    cfg = _make_config()
    cfg.rllm.algorithm.warmup_steps_ratio = 0.1
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.warmup_steps_ratio"})
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio == 0.1


def test_zero_rllm_warmup_steps_keeps_verl_ratio_fallback(propagate):
    cfg = _make_config()
    cfg.rllm.algorithm.warmup_steps = 0
    cfg.rllm.algorithm.warmup_steps_ratio = 0.1
    cfg = propagate(
        cfg,
        explicit_keys={
            "rllm.algorithm.warmup_steps",
            "rllm.algorithm.warmup_steps_ratio",
        },
    )
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio == 0.1


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


def test_verl_kl_loss_coef_propagates_to_rllm_kl_beta_with_warning(propagate, caplog):
    """User typed verl-side → kl_beta synced; use_kl_loss derived True."""
    caplog.set_level(logging.WARNING, logger="rllm.experimental.verl.utils")
    cfg = _make_config()
    cfg.actor_rollout_ref.actor.kl_loss_coef = 0.02
    cfg = propagate(cfg, explicit_keys={"actor_rollout_ref.actor.kl_loss_coef"})
    assert cfg.rllm.algorithm.kl_beta == 0.02
    assert cfg.actor_rollout_ref.actor.use_kl_loss is True
    assert "actor_rollout_ref.actor.kl_loss_coef is deprecated" in caplog.text


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
    assert cfg.actor_rollout_ref.actor.clip_ratio_low == 0.3
    assert cfg.actor_rollout_ref.actor.clip_ratio_high == 0.3


def test_rllm_eps_clip_high_sets_asymmetric_upper_bound(propagate):
    cfg = _make_config()
    cfg.rllm.algorithm.eps_clip = 0.2
    cfg.rllm.algorithm.eps_clip_high = 0.28
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.eps_clip", "rllm.algorithm.eps_clip_high"})
    assert cfg.actor_rollout_ref.actor.clip_ratio == 0.2
    assert cfg.actor_rollout_ref.actor.clip_ratio_low == 0.2
    assert cfg.actor_rollout_ref.actor.clip_ratio_high == 0.28


def test_verl_clip_ratio_propagates_to_rllm_eps_clip_with_warning(propagate, caplog):
    """User sets verl.clip_ratio (no asymmetric override) → rllm.eps_clip mirrors it."""
    caplog.set_level(logging.WARNING, logger="rllm.experimental.verl.utils")
    cfg = _make_config()
    cfg.actor_rollout_ref.actor.clip_ratio = 0.3
    cfg = propagate(cfg, explicit_keys={"actor_rollout_ref.actor.clip_ratio"})
    assert cfg.rllm.algorithm.eps_clip == 0.3
    assert "actor_rollout_ref.actor.clip_ratio is deprecated" in caplog.text


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


def test_rllm_eps_clip_wins_over_verl_clip_ratio_conflict(propagate, caplog):
    caplog.set_level(logging.WARNING, logger="rllm.experimental.verl.utils")
    cfg = _make_config()
    cfg.rllm.algorithm.eps_clip = 0.3
    cfg.actor_rollout_ref.actor.clip_ratio = 0.15
    cfg = propagate(cfg, explicit_keys={"rllm.algorithm.eps_clip", "actor_rollout_ref.actor.clip_ratio"})
    assert cfg.rllm.algorithm.eps_clip == 0.3
    assert cfg.actor_rollout_ref.actor.clip_ratio == 0.3
    assert cfg.actor_rollout_ref.actor.clip_ratio_low == 0.3
    assert "actor_rollout_ref.actor.clip_ratio conflicts with rllm.algorithm.eps_clip" in caplog.text


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
