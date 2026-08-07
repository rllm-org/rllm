"""Tinker SFT dataset, optimizer, scheduling, and fit-loop contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.tinker_backend import (  # noqa: E402
    TinkerSFTBackend,
    build_adam_params,
    iter_training_batches,
)
from rllm.trainer.sft.tinker_dataset import TinkerSFTDataset  # noqa: E402


class _TokenRenderer:
    """Small real Renderer boundary without a tokenizer dependency."""

    def build_supervised_example(self, messages, train_on_what):
        import torch

        n = int(messages[-1]["content"][0]["text"])
        return tinker.ModelInput.from_ints(list(range(n + 2))), torch.tensor(
            [0.0, *([1.0] * n), 0.0],
            dtype=torch.float32,
        )


def _length_dataset(lengths):
    return Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q", "trainable": False},
                    {"role": "assistant", "content": str(n), "trainable": True},
                ]
            }
            for n in lengths
        ],
        name="lengths",
        split="train",
    )


def test_token_mean_reduction_weights_every_assistant_token_equally():
    ds = TinkerSFTDataset(
        _length_dataset([2, 6]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
        loss_reduction="token_mean",
    )
    batch = ds.get_batch(0)
    weights = [d.loss_fn_inputs["weights"].data for d in batch]

    assert sum(sum(w) for w in weights) == pytest.approx(1.0)
    positive = [x for w in weights for x in w if x > 0]
    assert len(positive) == 8
    assert positive == pytest.approx([1 / 8] * 8)


def test_sequence_mean_reduction_gives_each_trajectory_equal_weight():
    ds = TinkerSFTDataset(
        _length_dataset([2, 6]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
        loss_reduction="sequence_mean",
    )
    weights = [d.loss_fn_inputs["weights"].data for d in ds.get_batch(0)]
    assert [sum(w) for w in weights] == pytest.approx([1.0, 1.0])


def test_dataset_rejects_nonpositive_batch_size():
    with pytest.raises(SFTConfigError, match="batch_size must be positive"):
        TinkerSFTDataset(
            _length_dataset([2]),
            renderer=_TokenRenderer(),
            batch_size=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_length": 1}, "max_length"),
        ({"overlength_policy": "drop"}, "overlength policy"),
        ({"loss_reduction": "batch_mean"}, "loss reduction"),
    ],
)
def test_dataset_rejects_invalid_render_settings_before_iteration(kwargs, match):
    with pytest.raises(SFTConfigError, match=match):
        TinkerSFTDataset(
            _length_dataset([2]),
            renderer=_TokenRenderer(),
            batch_size=1,
            **kwargs,
        )


def test_training_batch_iterator_caps_mid_epoch_at_exact_max_steps():
    batches = list(
        iter_training_batches(
            n_batches=3262,
            total_epochs=1,
            start_epoch=0,
            start_batch=0,
            max_steps=1000,
        )
    )
    assert len(batches) == 1000
    assert batches[0] == (0, 0, 0)
    assert batches[-1] == (999, 0, 999)


def test_training_batch_iterator_resumes_at_next_unseen_batch():
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


def test_adamw_options_reach_tinker_sdk():
    adam = build_adam_params(
        learning_rate=1e-4,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=1e-2,
        grad_clip_norm=1.0,
    )
    assert adam.learning_rate == pytest.approx(1e-4)
    assert adam.beta1 == pytest.approx(0.9)
    assert adam.beta2 == pytest.approx(0.999)
    assert adam.weight_decay == pytest.approx(1e-2)
    assert adam.grad_clip_norm == pytest.approx(1.0)


class _FakeDataset:
    def __init__(self, datums, batches):
        self.datums = datums
        self.batches = batches
        self.batch_calls = []
        self.epoch_seeds = []

    def __len__(self):
        return self.batches

    def get_batch(self, index):
        self.batch_calls.append(index)
        return self.datums

    def set_epoch(self, seed):
        self.epoch_seeds.append(seed)


class _FakeFuture:
    def __init__(self, value):
        self.value = value

    async def result_async(self):
        return self.value


class _FakeTrainingClient:
    def __init__(self, datums):
        self.datums = datums
        self.adam = []
        self.forward_calls = 0
        self.forward_backward_calls = 0

    def _forward_output(self):
        outputs = []
        for datum in self.datums:
            weights = datum.loss_fn_inputs["weights"]
            outputs.append(
                {
                    "logprobs": tinker.TensorData(
                        data=[-1.0] * len(weights.data),
                        dtype=weights.dtype,
                        shape=list(weights.shape),
                    )
                }
            )
        return SimpleNamespace(loss_fn_outputs=outputs)

    async def forward_async(self, data, loss_fn):
        assert data == self.datums
        assert loss_fn == "cross_entropy"
        self.forward_calls += 1
        return _FakeFuture(self._forward_output())

    async def forward_backward_async(self, data, loss_fn):
        assert data == self.datums
        assert loss_fn == "cross_entropy"
        self.forward_backward_calls += 1
        return _FakeFuture(self._forward_output())

    async def optim_step_async(self, adam):
        self.adam.append(adam)
        return _FakeFuture(SimpleNamespace(metrics={"optim/grad_norm": 0.5}))


class _FakeTracking:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logs = []
        self.finished = False
        self.instances.append(self)

    def log(self, data, step):
        self.logs.append((step, dict(data)))

    def finish(self):
        self.finished = True


def _one_datum():
    import torch
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    return datum_from_model_input_weights(
        tinker.ModelInput.from_ints([1, 2, 3, 4]),
        torch.tensor([0.0, 1.0, 1.0, 0.0]),
        max_length=None,
        reduction="none",
    )


def test_fit_loop_uses_completed_step_cadence_and_saves_resume_cursor(
    monkeypatch,
    tmp_path,
):
    """Exercise the real async loop boundary without opening a provider job."""
    from tinker_cookbook import checkpoint_utils, display

    import rllm.trainer.sft.tinker_backend as backend_module
    import rllm.utils.tracking as tracking_module

    datum = _one_datum()
    train = _FakeDataset([datum], batches=3)
    val = _FakeDataset([datum], batches=1)
    client = _FakeTrainingClient([datum])
    saves = []

    class _ServiceClient:
        def __init__(self, base_url=None):
            self.base_url = base_url

        async def create_lora_training_client_async(self, **kwargs):
            return client

    async def _save(**kwargs):
        saves.append(kwargs)
        return {}

    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (object(), train, val))
    monkeypatch.setattr(tinker, "ServiceClient", _ServiceClient)
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: None)
    monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", _save)
    monkeypatch.setattr(display, "colorize_example", lambda *_: "example")
    monkeypatch.setattr(tracking_module, "Tracking", _FakeTracking)
    _FakeTracking.instances.clear()

    spec = SFTSpec(
        train_dataset=_length_dataset([2]),
        val_dataset=_length_dataset([2]),
        output_dir=str(tmp_path),
        overrides={
            "trainer": {
                "total_epochs": 2,
                "max_steps": 3,
                "save_freq": 2,
                "test_freq": 2,
            },
            "optim": {
                "lr": 1e-4,
                "min_lr": 1e-5,
                "lr_scheduler": "cosine",
                "warmup_steps": 1,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 1e-2,
                "grad_clip_norm": 1.0,
            },
        },
    )
    backend = TinkerSFTBackend(spec)
    backend.build_config()
    asyncio.run(backend._fit_async())

    assert train.batch_calls == [0, 1, 2]
    assert train.epoch_seeds == [0]
    assert client.forward_backward_calls == 3
    assert client.forward_calls == 2  # validation at step 0 and completed step 2
    assert [save["name"] for save in saves] == ["000002", "final"]
    assert saves[0]["loop_state"] == {"epoch": 0, "batch": 2, "step": 2}
    assert saves[1]["loop_state"] == {
        "epoch": 1,
        "batch": 0,
        "step": 3,
        "final": True,
    }
    assert len(client.adam) == 3
    assert all(adam.weight_decay == pytest.approx(1e-2) for adam in client.adam)
    tracking = _FakeTracking.instances[-1]
    assert [step for step, _ in tracking.logs] == [0, 1, 2, 3, 3]
    assert tracking.logs[-1][1] == {"status": "completed"}
    assert tracking.finished is True
