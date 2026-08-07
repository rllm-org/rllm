"""SFT learning-rate schedule and optimizer configuration.

The SFT fits multiply the base LR by ``sft_lr_multiplier`` each step. Paper-
aligned agentic SFT uses a linear warmup (ratio 0.1) then a cosine decay; the
tinker_cookbook multiplier the fits used before had no warmup term, so the
``warmup_steps_ratio`` config knob was dead. These pin the warmup math and that
``warmup_steps_ratio=0`` is an exact no-op (unchanged default behavior).
"""

from __future__ import annotations

import pytest

pytest.importorskip("tinker_cookbook")  # sft_lr_multiplier delegates decay to it

from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier as _cookbook  # noqa: E402

from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.tinker_backend import (  # noqa: E402
    resolve_sft_optimizer_settings,
    sft_lr_multiplier,
)


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


@pytest.mark.parametrize("schedule", ["cosine", "linear", "constant"])
def test_zero_warmup_reduces_to_cookbook(schedule):
    for step in (0, 25, 50, 99):
        assert sft_lr_multiplier(schedule, step, 100, warmup_steps_ratio=0.0) == pytest.approx(_cookbook(lr_schedule=schedule, step=step, total_steps=100))


def test_cosine_and_warmup_reach_fireworks_subclass_config():
    """cosine + warmup must land in the FIREWORKS config (the subclass path the
    real run uses), not just tinker's."""
    from rllm.data import Dataset
    from rllm.trainer.sft import SFTSpec
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    ds = Dataset(data=[{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}], name="t", split="train")
    spec = SFTSpec(model="Qwen/Qwen2.5-7B-Instruct", train_dataset=ds, lr_schedule="cosine", overrides={"optim": {"warmup_steps_ratio": 0.1}})
    cfg = FireworksSFTBackend(spec).build_config()
    assert cfg.optim.lr_scheduler == "cosine"
    assert cfg.optim.warmup_steps_ratio == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"lr": 0}, "optim.lr must be positive"),
        ({"lr": 1e-4, "min_lr": 2e-4}, "optim.min_lr"),
        ({"lr_scheduler": "polynomial"}, "optim.lr_scheduler"),
        ({"warmup_steps_ratio": 1.5}, "warmup_steps_ratio"),
        ({"warmup_steps": 101}, "cannot exceed"),
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
