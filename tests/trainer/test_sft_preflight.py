"""Hosted SFT data is rendered before a provider is started."""

from __future__ import annotations

import asyncio
import sys
import types

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
    def __init__(self):
        self.seen: list[int] = []

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        from rllm.renderers.types import RenderedTokens

        del tools, add_generation_prompt
        count = int(messages[-1]["content"])
        self.seen.append(count)
        return RenderedTokens(
            token_ids=list(range(count + 2)),
            message_indices=[-1, *([1] * count), -1],
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


def test_preflight_renders_the_exact_planned_training_order():
    source = _dataset([2, 3, 4, 5])
    renderer = _TokenRenderer()
    dataset = TinkerSFTDataset(source, renderer=renderer, batch_size=2)
    plan = [(0, 0), (0, 1), (1, 0)]

    dataset.preflight(label="train", planned_batches=plan)
    preflight_order = renderer.seen.copy()
    assert dataset.dataset is source

    renderer.seen.clear()
    current_epoch = None
    for epoch_idx, batch_idx in plan:
        if epoch_idx != current_epoch:
            dataset.set_epoch(seed=epoch_idx)
            current_epoch = epoch_idx
        dataset.get_batch(batch_idx)

    assert renderer.seen == preflight_order


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
    training = types.ModuleType("training")
    training_utils = types.ModuleType("training.utils")
    checkpoints = types.ModuleType("training.utils.checkpoints")
    checkpoints.TrainingCheckpoints = object
    client = types.ModuleType("training.utils.client")
    client.DEFAULT_TIMEOUT_S = 1
    monkeypatch.setitem(sys.modules, "training", training)
    monkeypatch.setitem(sys.modules, "training.utils", training_utils)
    monkeypatch.setitem(sys.modules, "training.utils.checkpoints", checkpoints)
    monkeypatch.setitem(sys.modules, "training.utils.client", client)
    monkeypatch.setattr(tracking_module, "Tracking", _Tracking)
    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (None, train, None))
    monkeypatch.setattr(backend, "_provision", lambda *_: pytest.fail("provider trainer must not be provisioned"))

    with pytest.raises(SFTConfigError, match="train preflight.*epoch 0.*batch 0"):
        backend.fit()
