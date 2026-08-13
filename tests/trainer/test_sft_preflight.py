"""Hosted SFT data is rendered before a provider is started."""

from __future__ import annotations

import asyncio

import pytest

tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend  # noqa: E402
from rllm.trainer.sft.tinker_backend import TinkerSFTBackend  # noqa: E402
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


def test_preflight_checks_planned_shuffle_and_restores_source_order():
    source = _dataset([0, 2, 0, 2])
    dataset = TinkerSFTDataset(source, renderer=_TokenRenderer(), batch_size=2)

    with pytest.raises(SFTConfigError, match="train preflight.*epoch 0.*batch 0"):
        dataset.preflight(label="train", planned_batches=[(0, 0), (0, 1)])

    assert dataset.dataset is source


def test_validation_preflight_checks_every_batch():
    dataset = TinkerSFTDataset(
        _dataset([2, 20]),
        renderer=_TokenRenderer(),
        batch_size=1,
        max_length=10,
        overlength_policy="error",
    )

    with pytest.raises(SFTConfigError, match="validation preflight.*batch 1.*max_length=10"):
        dataset.preflight(label="validation")


def test_tinker_preflight_failure_precedes_service_client(monkeypatch, tmp_path):
    from tinker_cookbook import checkpoint_utils

    import rllm.trainer.sft.tinker_backend as backend_module

    train = TinkerSFTDataset(
        _dataset([0, 2, 0, 2]),
        renderer=_TokenRenderer(),
        batch_size=2,
    )
    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (object(), train, None))
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: None)
    monkeypatch.setattr(tinker, "ServiceClient", lambda **_: pytest.fail("provider client must not be created"))

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_dataset([2]),
            output_dir=str(tmp_path),
            batch_size=2,
            overrides={"trainer": {"max_steps": 2}},
        )
    )
    backend.build_config()

    with pytest.raises(SFTConfigError, match="train preflight.*epoch 0.*batch 0"):
        asyncio.run(backend._fit_async())


def test_fireworks_preflight_failure_precedes_provision(monkeypatch, tmp_path):
    import rllm.trainer.sft.fireworks_backend as backend_module
    import rllm.utils.tracking as tracking_module

    train = TinkerSFTDataset(
        _dataset([0, 2, 0, 2]),
        renderer=_TokenRenderer(),
        batch_size=2,
    )
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=_dataset([2]),
            output_dir=str(tmp_path),
            batch_size=2,
            overrides={"trainer": {"max_steps": 2}},
        )
    )
    backend.build_config()

    class _Tracking:
        def __init__(self, **_kwargs):
            pass

        def finish(self):
            pass

    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    monkeypatch.setattr(tracking_module, "Tracking", _Tracking)
    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (None, train, None))
    monkeypatch.setattr(backend, "_provision", lambda *_: pytest.fail("provider trainer must not be provisioned"))

    with pytest.raises(SFTConfigError, match="train preflight.*epoch 0.*batch 0"):
        backend.fit()
