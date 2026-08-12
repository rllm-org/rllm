"""Fireworks SFT cursor, cadence, and request-ordering contracts.

The loop runs entirely against fakes: no provider resources are created and no
API calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rllm.data import Dataset
from rllm.trainer.sft import SFTSpec
from rllm.trainer.sft.backend import SFTConfigError
from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend
from rllm.trainer.sft.tinker_dataset import TinkerSFTDataset


class _Future:
    def __init__(self, events, event, result):
        self.events = events
        self.event = event
        self.value = result

    def result(self, timeout=None):
        del timeout
        self.events.append(self.event)
        return self.value


class _Client:
    def __init__(self, events):
        self.events = events

    def submit_forward_backward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        batch = data[0]
        self.events.append(f"submit-fb-{batch}")
        result = SimpleNamespace(metrics={"response_tokens": 2, "loss:sum": 2.0})
        return _Future(self.events, f"finish-fb-{batch}", result)

    def submit_optim_step(self, adam):
        del adam
        self.events.append("submit-opt")
        return _Future(self.events, "finish-opt", SimpleNamespace())


class _Dataset:
    def __init__(self, events, n_batches):
        self.events = events
        self.n_batches = n_batches
        self.preflighting = False

    def __len__(self):
        return self.n_batches

    def set_epoch(self, seed):
        prefix = "preflight-" if self.preflighting else ""
        self.events.append(f"{prefix}epoch-{seed}")

    def get_batch(self, index):
        prefix = "preflight-batch" if self.preflighting else "batch"
        self.events.append(f"{prefix}-{index}")
        return [index]

    def preflight(self, label="train", planned_batches=None):
        self.events.append(f"preflight-{label}")
        self.preflighting = True
        try:
            for index in range(len(self)):
                self.get_batch(index)
            batches = planned_batches or ()
            current_epoch = None
            for epoch, index in batches:
                if epoch is not None and epoch != current_epoch:
                    self.set_epoch(epoch)
                    current_epoch = epoch
                self.get_batch(index)
        finally:
            self.preflighting = False

    def data_cursor_for_step(self, completed_steps):
        return completed_steps

    def step_for_data_cursor(self, data_consumed):
        return data_consumed


class _Tracking:
    def __init__(self, events, **kwargs):
        del kwargs
        self.events = events

    def log(self, data, step):
        self.events.append(f"log-{step}-{data.get('status', 'metrics')}")

    def finish(self):
        self.events.append("tracking-finish")


class _Checkpoints:
    def __init__(self, events, resume_step, resume_checkpoint_step):
        self.events = events
        self.resume_step = resume_step
        self.resume_checkpoint_step = resume_checkpoint_step
        self.saves = []

    def resume(self):
        self.events.append("resume")
        if self.resume_step is None:
            return None
        # Simulate a provider-renamed checkpoint whose name-derived step is
        # unusable; the separately persisted data cursor remains authoritative.
        return SimpleNamespace(
            step=self.resume_checkpoint_step,
            data_consumed=self.resume_step,
        )

    def save(self, name, **kwargs):
        self.saves.append((name, kwargs))
        self.events.append(f"save-{name}")

    def promote_latest(self, output_model_id, base_model):
        del base_model
        self.events.append("promote")
        return {"name": output_model_id}


def _run_fit(
    monkeypatch,
    tmp_path,
    *,
    resume_step=None,
    save_freq=2,
    test_freq=2,
    events=None,
    resume_checkpoint_step=0,
):
    import training.utils.checkpoints as checkpoints_module

    import rllm.trainer.sft.fireworks_backend as module
    import rllm.utils.tracking as tracking_module

    events = [] if events is None else events
    train_dataset = _Dataset(events, n_batches=3)
    val_dataset = _Dataset(events, n_batches=1)
    client = _Client(events)
    checkpoints = _Checkpoints(events, resume_step, resume_checkpoint_step)
    infra = SimpleNamespace(
        policy=client,
        service=object(),
        policy_job_id="fake-job",
        close=lambda: events.append("infra-close"),
    )

    source = Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ],
        name="fake",
        split="train",
    )
    backend = FireworksSFTBackend(
        SFTSpec(
            model="Qwen/Qwen3-0.6B",
            train_dataset=source,
            val_dataset=source,
            epochs=1,
            output_dir=str(tmp_path),
        )
    )
    config = backend.build_config()
    config.trainer.max_steps = 3
    config.trainer.save_freq = save_freq
    config.trainer.test_freq = test_freq

    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    monkeypatch.setattr(
        module,
        "build_sft_data",
        lambda config, train, val: (None, train_dataset, val_dataset),
    )
    monkeypatch.setattr(
        backend,
        "_provision",
        lambda config, api_key, base_url: events.append("provision") or infra,
    )
    monkeypatch.setattr(
        backend,
        "_validate",
        lambda client, dataset: events.append("validate") or {"test/loss": 1.0},
    )
    monkeypatch.setattr(
        tracking_module,
        "Tracking",
        lambda **kwargs: _Tracking(events, **kwargs),
    )
    monkeypatch.setattr(
        checkpoints_module,
        "TrainingCheckpoints",
        lambda *args, **kwargs: checkpoints,
    )

    backend.fit()
    return events, checkpoints


def test_fireworks_validation_and_checkpoint_follow_completed_update(monkeypatch, tmp_path):
    events, checkpoints = _run_fit(monkeypatch, tmp_path)

    assert events.index("preflight-validation") < events.index("provision")

    # Ordinary updates keep one later update submitted while the prior result
    # is collected.
    assert events.index("submit-fb-1") < events.index("finish-fb-0")

    assert events.count("validate") == 2  # initial step 0, then completed step 2
    second_validate = [i for i, event in enumerate(events) if event == "validate"][1]
    assert events.index("finish-opt", events.index("batch-1")) < second_validate
    assert second_validate < events.index("batch-2")
    assert checkpoints.saves == [
        (
            "step-2",
            {
                "resumable": True,
                "promotable": False,
                "data_consumed": 2,
            },
        ),
        (
            "step-3",
            {
                "resumable": True,
                "promotable": True,
                "data_consumed": 3,
            },
        ),
    ]


def test_fireworks_resume_uses_checkpoint_as_next_batch_cursor(monkeypatch, tmp_path):
    events, checkpoints = _run_fit(monkeypatch, tmp_path, resume_step=2)

    assert "batch-0" not in events
    assert "batch-1" not in events
    assert events.count("batch-2") == 1
    assert "epoch-0" in events  # reconstruct the epoch's deterministic ordering
    assert "validate" not in events  # initial validation is fresh-run only
    assert [name for name, _ in checkpoints.saves] == ["step-3"]


def test_fireworks_resume_rejects_missing_persisted_cursor_before_training(
    monkeypatch,
    tmp_path,
):
    events = []

    with pytest.raises(SFTConfigError, match="without a positive persisted dataset cursor"):
        _run_fit(
            monkeypatch,
            tmp_path,
            resume_step=0,
            resume_checkpoint_step=5,
            events=events,
        )

    assert not any(event.startswith("submit-fb-") for event in events)


@pytest.mark.parametrize("data_consumed", [-1, 4])
def test_fireworks_resume_rejects_invalid_data_cursor(
    monkeypatch,
    tmp_path,
    data_consumed,
):
    events = []

    with pytest.raises(SFTConfigError, match="cursor|horizon"):
        _run_fit(
            monkeypatch,
            tmp_path,
            resume_step=data_consumed,
            events=events,
        )

    assert not any(event.startswith("submit-fb-") for event in events)


def test_fireworks_normalizes_quoted_checkpoint_cadence_before_provision(
    monkeypatch,
    tmp_path,
):
    events, checkpoints = _run_fit(
        monkeypatch,
        tmp_path,
        save_freq="2",
        test_freq="2",
    )

    assert events.index("preflight-train") < events.index("provision")
    assert [name for name, _ in checkpoints.saves] == ["step-2", "step-3"]


@pytest.mark.parametrize("setting", ["save_freq", "test_freq"])
def test_fireworks_rejects_malformed_cadence_before_provision(
    monkeypatch,
    tmp_path,
    setting,
):
    events = []
    kwargs = {setting: "every-two-steps"}

    with pytest.raises(SFTConfigError, match=rf"trainer.{setting}.*integer"):
        _run_fit(monkeypatch, tmp_path, events=events, **kwargs)

    assert "provision" not in events


class _MaskRenderer:
    def build_supervised_example(self, messages, train_on_what):
        del train_on_what
        import tinker
        import torch

        n = int(messages[-1]["content"][0]["text"])
        return tinker.ModelInput.from_ints(list(range(n + 2))), torch.tensor(
            [0.0, *([1.0] * n), 0.0],
            dtype=torch.float32,
        )


def test_fireworks_preflights_the_shuffled_epoch_order_before_provision(
    monkeypatch,
    tmp_path,
):
    import rllm.trainer.sft.fireworks_backend as module

    source = Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q", "trainable": False},
                    {"role": "assistant", "content": str(n), "trainable": True},
                ]
            }
            for n in [0, 2, 0, 2]
        ],
        name="shuffled-mask",
        split="train",
    )
    train_dataset = TinkerSFTDataset(
        source,
        renderer=_MaskRenderer(),
        batch_size=2,
        max_length=100,
    )
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=source,
            output_dir=str(tmp_path),
            batch_size=2,
            overrides={"trainer": {"max_steps": 2}},
        )
    )
    backend.build_config()
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    monkeypatch.setattr(
        module,
        "build_sft_data",
        lambda *_: (None, train_dataset, None),
    )
    monkeypatch.setattr(
        backend,
        "_provision",
        lambda *_: pytest.fail("provider trainer must not be provisioned"),
    )

    with pytest.raises(SFTConfigError, match="train preflight.*epoch 0.*batch 0"):
        backend.fit()


def test_fireworks_bad_preflight_never_provisions(monkeypatch, tmp_path):
    import rllm.trainer.sft.fireworks_backend as module

    events = []
    train_dataset = _Dataset(events, n_batches=2)

    def fail_preflight(label, planned_batches=None):
        del planned_batches
        raise module.SFTConfigError(f"{label} row is invalid")

    train_dataset.preflight = fail_preflight
    source = Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ],
        name="invalid-preflight",
        split="train",
    )
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=source,
            output_dir=str(tmp_path),
            overrides={"trainer": {"max_steps": 1}},
        )
    )
    backend.build_config()
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    monkeypatch.setattr(
        module,
        "build_sft_data",
        lambda *_: (None, train_dataset, None),
    )
    monkeypatch.setattr(
        backend,
        "_provision",
        lambda *_: pytest.fail("provider trainer must not be provisioned"),
    )

    with pytest.raises(module.SFTConfigError, match="train row is invalid"):
        backend.fit()
