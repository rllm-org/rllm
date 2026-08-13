"""Focused contracts for hosted-SFT batching and loss semantics."""

from __future__ import annotations

import pytest

tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.tinker_dataset import TinkerSFTDataset  # noqa: E402


class _TokenRenderer:
    def build_supervised_example(self, messages, train_on_what):
        import torch

        del train_on_what
        count = int(messages[-1]["content"][0]["text"])
        return tinker.ModelInput.from_ints(list(range(count + 2))), torch.tensor(
            [0.0, *([1.0] * count), 0.0],
            dtype=torch.float32,
        )


def _dataset(lengths):
    return Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q", "trainable": False},
                    {"role": "assistant", "content": str(length), "trainable": True},
                ]
            }
            for length in lengths
        ],
        name="lengths",
        split="train",
    )


def test_token_mean_weights_every_supervised_token_equally():
    dataset = TinkerSFTDataset(
        _dataset([2, 6]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
        loss_reduction="token_mean",
    )
    weights = [datum.loss_fn_inputs["weights"].data for datum in dataset.get_batch(0)]
    positive = [weight for row in weights for weight in row if weight > 0]
    assert positive == pytest.approx([1 / 8] * 8)


def test_sequence_mean_gives_each_trajectory_equal_weight():
    dataset = TinkerSFTDataset(
        _dataset([2, 6]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
        loss_reduction="sequence_mean",
    )
    weights = [datum.loss_fn_inputs["weights"].data for datum in dataset.get_batch(0)]
    assert [sum(row) for row in weights] == pytest.approx([1.0, 1.0])


def test_default_loss_reduction_preserves_raw_weights():
    dataset = TinkerSFTDataset(
        _dataset([2, 6]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
    )
    weights = [datum.loss_fn_inputs["weights"].data for datum in dataset.get_batch(0)]
    assert [sum(row) for row in weights] == pytest.approx([2.0, 6.0])


def test_final_partial_batch_is_not_dropped():
    dataset = TinkerSFTDataset(
        _dataset([2, 2, 2]),
        renderer=_TokenRenderer(),
        batch_size=2,
    )
    assert len(dataset) == 2
    assert len(dataset.get_batch(1)) == 1


def test_overlength_error_is_explicit_opt_in():
    dataset = TinkerSFTDataset(
        _dataset([20]),
        renderer=_TokenRenderer(),
        batch_size=1,
        max_length=10,
        overlength_policy="error",
    )
    with pytest.raises(SFTConfigError, match="trajectory was not truncated"):
        dataset.get_batch(0)


def test_batch_without_supervised_tokens_is_rejected():
    dataset = TinkerSFTDataset(
        _dataset([0]),
        renderer=_TokenRenderer(),
        batch_size=1,
    )
    with pytest.raises(SFTConfigError, match="no trainable tokens"):
        dataset.get_batch(0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": 1, "max_length": 1}, "max_length"),
        ({"batch_size": 1, "overlength_policy": "drop"}, "overlength policy"),
        ({"batch_size": 1, "loss_reduction": "batch_mean"}, "loss reduction"),
    ],
)
def test_invalid_render_settings_fail_before_iteration(kwargs, message):
    with pytest.raises(SFTConfigError, match=message):
        TinkerSFTDataset(_dataset([2]), renderer=_TokenRenderer(), **kwargs)
