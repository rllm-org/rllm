"""Fireworks SFT cursor, cadence, and request-ordering contracts.

The loop runs entirely against fakes: no provider resources are created and no
API calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace

from rllm.data import Dataset
from rllm.trainer.sft import SFTSpec
from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend


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

    def __len__(self):
        return self.n_batches

    def set_epoch(self, seed):
        self.events.append(f"epoch-{seed}")

    def get_batch(self, index):
        self.events.append(f"batch-{index}")
        return [index]

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
    def __init__(self, events, resume_step):
        self.events = events
        self.resume_step = resume_step
        self.saves = []

    def resume(self):
        self.events.append("resume")
        if self.resume_step is None:
            return None
        # Simulate a provider-renamed checkpoint whose name-derived step is
        # unusable; the separately persisted data cursor remains authoritative.
        return SimpleNamespace(step=0, data_consumed=self.resume_step)

    def save(self, name, **kwargs):
        self.saves.append((name, kwargs))
        self.events.append(f"save-{name}")

    def promote_latest(self, output_model_id, base_model):
        del base_model
        self.events.append("promote")
        return {"name": output_model_id}


def _run_fit(monkeypatch, tmp_path, *, resume_step=None):
    import training.utils.checkpoints as checkpoints_module

    import rllm.trainer.sft.fireworks_backend as module
    import rllm.utils.tracking as tracking_module

    events = []
    train_dataset = _Dataset(events, n_batches=3)
    val_dataset = object()
    client = _Client(events)
    checkpoints = _Checkpoints(events, resume_step)
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
    config.trainer.save_freq = 2
    config.trainer.test_freq = 2

    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    monkeypatch.setattr(
        module,
        "build_sft_data",
        lambda config, train, val: (None, train_dataset, val_dataset),
    )
    monkeypatch.setattr(backend, "_provision", lambda config, api_key, base_url: infra)
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
