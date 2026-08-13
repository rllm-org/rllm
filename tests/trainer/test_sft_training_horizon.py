"""Exact hosted-SFT training horizon contracts."""

import pytest

from rllm.trainer.sft.backend import SFTConfigError
from rllm.trainer.sft.tinker_backend import iter_training_batches, resolve_training_steps


def test_training_iterator_caps_mid_epoch_at_exact_max_steps():
    batches = list(
        iter_training_batches(
            n_batches=3262,
            total_epochs=1,
            max_steps=1000,
        )
    )
    assert len(batches) == 1000
    assert batches[0] == (0, 0, 0)
    assert batches[-1] == (999, 0, 999)


def test_training_iterator_honors_a_caller_supplied_start_cursor():
    batches = list(
        iter_training_batches(
            n_batches=20,
            total_epochs=2,
            start_epoch=1,
            start_batch=3,
            max_steps=30,
        )
    )
    assert batches[0] == (23, 1, 3)
    assert batches[-1] == (29, 1, 9)


def test_max_steps_cannot_extend_available_training():
    assert resolve_training_steps(3, 2, None) == 6
    assert resolve_training_steps(3, 2, 100) == 6


@pytest.mark.parametrize(
    ("n_batches", "epochs", "max_steps", "message"),
    [
        (0, 1, None, "no batches"),
        (1, 0, None, "total_epochs"),
        (1, True, None, "total_epochs"),
        (1, 1, 0, "max_steps"),
        (1, 1, True, "max_steps"),
    ],
)
def test_invalid_training_horizons_fail_before_launch(n_batches, epochs, max_steps, message):
    with pytest.raises(SFTConfigError, match=message):
        resolve_training_steps(n_batches, epochs, max_steps)
