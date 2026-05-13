import pytest

from rllm.trainer.tinker.tinker_policy_trainer import compute_schedule_lr_multiplier


def test_unset_warmup_steps_uses_warmup_ratio():
    multiplier = compute_schedule_lr_multiplier(
        lr_schedule="constant",
        warmup_steps_ratio=0.5,
        step=5,
        total_steps=20,
    )

    assert multiplier == pytest.approx(0.5)


def test_absolute_warmup_steps_override_warmup_ratio():
    multiplier = compute_schedule_lr_multiplier(
        lr_schedule="constant",
        warmup_steps_ratio=0.5,
        step=5,
        total_steps=100,
        warmup_steps=10,
    )

    assert multiplier == pytest.approx(0.5)


def test_zero_absolute_warmup_uses_warmup_ratio():
    multiplier = compute_schedule_lr_multiplier(
        lr_schedule="constant",
        warmup_steps_ratio=0.5,
        step=0,
        total_steps=10,
        warmup_steps=0,
    )

    assert multiplier == pytest.approx(0.0)


def test_warmup_steps_equal_total_steps_stays_finite_at_boundary():
    multiplier = compute_schedule_lr_multiplier(
        lr_schedule="linear",
        warmup_steps_ratio=0.0,
        step=10,
        total_steps=10,
        warmup_steps=10,
    )

    assert multiplier == pytest.approx(1.0)
