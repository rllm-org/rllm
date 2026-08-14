"""SFT learning-rate schedule and optimizer configuration.

The SFT fits multiply the base LR by ``sft_lr_multiplier`` each step. Paper-
aligned agentic SFT uses a linear warmup (ratio 0.1) then a cosine decay; the
tinker_cookbook multiplier the fits used before had no warmup term, so the
``warmup_steps_ratio`` config knob was dead. These pin the warmup math and that
``warmup_steps_ratio=0`` remains an exact opt-out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip("tinker_cookbook")  # sft_lr_multiplier delegates decay to it

from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier as _cookbook  # noqa: E402

from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.tinker_backend import (  # noqa: E402
    build_adam_params,
    resolve_sft_optimizer_settings,
    sft_lr_multiplier,
)


@pytest.mark.parametrize("backend", ["tinker", "fireworks"])
def test_hosted_configs_preserve_ten_percent_warmup_default(backend):
    config_path = Path(__file__).resolve().parents[2] / "rllm" / "trainer" / "sft" / "config" / f"{backend}.yaml"
    config = OmegaConf.load(config_path)
    assert config.optim.warmup_steps_ratio == pytest.approx(0.1)


def test_warmup_ramps_linearly_then_cosine_decays():
    # 10% of 100 steps => 10 warmup steps: linear 0->1, then cosine over the rest.
    assert sft_lr_multiplier("cosine", 0, 100, warmup_steps_ratio=0.1) == pytest.approx(0.0)
    assert sft_lr_multiplier("cosine", 5, 100, warmup_steps_ratio=0.1) == pytest.approx(0.5)
    assert sft_lr_multiplier("cosine", 10, 100, warmup_steps_ratio=0.1) == pytest.approx(1.0)  # cosine(0)
    assert sft_lr_multiplier("cosine", 100, 100, warmup_steps_ratio=0.1) == pytest.approx(0.0)  # cosine(pi)


def test_cosine_decay_respects_minimum_lr_ratio():
    """A configured 10% floor decays to that floor, not to zero."""
    assert sft_lr_multiplier(
        "cosine",
        100,
        100,
        min_lr_ratio=0.1,
    ) == pytest.approx(0.1)
    assert sft_lr_multiplier(
        "cosine",
        50,
        100,
        min_lr_ratio=0.1,
    ) == pytest.approx(0.55)


def test_invalid_minimum_lr_ratio_fails_during_warmup():
    with pytest.raises(SFTConfigError, match="must be in"):
        sft_lr_multiplier(
            "cosine",
            step=0,
            total_steps=100,
            warmup_steps=10,
            min_lr_ratio=1.1,
        )


def test_absolute_warmup_steps_override_ratio():
    # warmup_steps=4 wins over the ratio: LR is half-ramped at step 2.
    assert sft_lr_multiplier("constant", 2, 100, warmup_steps_ratio=0.5, warmup_steps=4) == pytest.approx(0.5)
    assert sft_lr_multiplier("constant", 4, 100, warmup_steps_ratio=0.5, warmup_steps=4) == pytest.approx(1.0)


def test_explicit_zero_warmup_overrides_nonzero_ratio():
    assert sft_lr_multiplier("constant", 0, 100, warmup_steps_ratio=0.5, warmup_steps=0) == pytest.approx(1.0)


def test_adam_controls_reach_the_provider_request():
    params = build_adam_params(
        learning_rate=2e-5,
        betas=(0.8, 0.9),
        eps=1e-7,
        weight_decay=0.01,
        grad_clip_norm=1.0,
    )

    assert params.learning_rate == pytest.approx(2e-5)
    assert params.beta1 == pytest.approx(0.8)
    assert params.beta2 == pytest.approx(0.9)
    assert params.eps == pytest.approx(1e-7)
    assert params.weight_decay == pytest.approx(0.01)
    assert params.grad_clip_norm == pytest.approx(1.0)


@pytest.mark.parametrize("schedule", ["cosine", "linear", "constant"])
def test_zero_warmup_reduces_to_cookbook(schedule):
    for step in (0, 25, 50, 99):
        assert sft_lr_multiplier(schedule, step, 100, warmup_steps_ratio=0.0) == pytest.approx(_cookbook(lr_schedule=schedule, step=step, total_steps=100))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"lr": 0}, "optim.lr must be positive"),
        ({"lr": 1e-4, "min_lr": 2e-4}, "optim.min_lr"),
        ({"lr_scheduler": "polynomial"}, "optim.lr_scheduler"),
        ({"warmup_steps_ratio": 1.5}, "warmup_steps_ratio"),
        ({"warmup_steps": 101}, "cannot exceed"),
        ({"warmup_steps": -2}, "warmup_steps"),
        ({"warmup_steps": 1.5}, "warmup_steps"),
        ({"betas": [0.9, 1.0]}, "optim.betas"),
        ({"eps": 0}, "optim.eps"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"grad_clip_norm": -1}, "grad_clip_norm"),
    ],
)
def test_optimizer_settings_reject_invalid_runs_before_provision(overrides, match):
    config = {
        "lr": 1e-4,
        "min_lr": 1e-5,
        "lr_scheduler": "cosine",
        "warmup_steps_ratio": 0.0,
        "warmup_steps": 10,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "grad_clip_norm": 1.0,
        **overrides,
    }
    with pytest.raises(SFTConfigError, match=match):
        resolve_sft_optimizer_settings(config, total_steps=100)
