"""In-loop SFT validation on the Tinker/Fireworks family.

Both backends accumulate gradients server-side across ``forward_backward``
calls until the next ``optim_step``, so a validation pass that runs backward
silently trains on the held-out set. These tests pin the val loop to the
forward-only API, pin the metric it reports, and pin the ``fit`` call sites —
all on fakes, so no Tinker/Fireworks spend.

The two backends share the assertions but not the call protocol: tinker's
``_validate`` is async over ``forward_async`` futures, fireworks' is sync over
``ReconnectableClient.forward`` (which blocks internally).
"""

import asyncio
from types import SimpleNamespace

import pytest

tinker = pytest.importorskip("tinker")
from tinker_cookbook.supervised.common import compute_mean_nll  # noqa: E402

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend  # noqa: E402
from rllm.trainer.sft.tinker_backend import TinkerSFTBackend, should_validate_step  # noqa: E402


@pytest.mark.parametrize(
    ("completed_steps", "include_initial", "expected"),
    [(0, False, False), (0, True, True), (1, False, False), (10, False, True), (20, False, True)],
)
def test_validation_cadence_uses_completed_steps(completed_steps, include_initial, expected):
    assert (
        should_validate_step(
            completed_steps,
            eval_every=10,
            has_validation=True,
            include_initial=include_initial,
        )
        is expected
    )


def test_validation_cadence_requires_validation_data():
    assert not should_validate_step(0, eval_every=10, has_validation=False, include_initial=True)
    assert not should_validate_step(10, eval_every=10, has_validation=False)


@pytest.mark.parametrize("eval_every", [0, -1])
@pytest.mark.parametrize(("completed_steps", "include_initial"), [(0, True), (10, False)])
def test_nonpositive_cadence_disables_all_validation(eval_every, completed_steps, include_initial):
    assert not should_validate_step(
        completed_steps,
        eval_every=eval_every,
        has_validation=True,
        include_initial=include_initial,
    )


def _tensor(values):
    import torch

    return tinker.TensorData.from_torch(torch.tensor(values, dtype=torch.float32))


class _Datum:
    """Only the fields ``_validate`` reads off a tinker Datum."""

    def __init__(self, weights, label=None):
        self.loss_fn_inputs = {"weights": _tensor(weights)}
        self.model_input = SimpleNamespace(length=len(weights))
        self.label = label


class _Output:
    """A forward / forward_backward response: per-token logprobs per datum."""

    def __init__(self, logprobs_per_datum):
        self.loss_fn_outputs = [{"logprobs": _tensor(lp)} for lp in logprobs_per_datum]


class _ValDataset:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def get_batch(self, idx):
        return self._batches[idx]


class _AsyncFuture:
    def __init__(self, output):
        self._output = output

    async def result_async(self):
        return self._output


class _SyncFuture:
    def __init__(self, output):
        self._output = output

    def result(self, timeout=None):
        return self._output


class _FakeTinkerClient:
    """Tinker training client logging (pass, loss_fn) per call.

    Exhausting ``outputs`` raises rather than replaying, so a val loop that
    calls more times than there are batches fails loudly.
    """

    def __init__(self, outputs):
        self.calls: list[tuple[str, str]] = []
        self._outputs = iter(outputs)

    async def forward_async(self, data, loss_fn):
        self.calls.append(("forward", loss_fn))
        return _AsyncFuture(next(self._outputs))

    async def forward_backward_async(self, data, loss_fn):
        self.calls.append(("forward_backward", loss_fn))
        return _AsyncFuture(next(self._outputs))


class _FakeFireworksClient:
    """``ReconnectableClient`` surface, logging (pass, loss_fn) per call."""

    def __init__(self, outputs):
        self.calls: list[tuple[str, str]] = []
        self._outputs = iter(outputs)

    def forward(self, data, loss_fn):
        self.calls.append(("forward", loss_fn))
        return next(self._outputs)

    def submit_forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
        self.calls.append(("forward_backward", loss_fn))
        return _SyncFuture(next(self._outputs))


# Two batches with different loss-token counts, so a token-weighted mean
# (2*1.5 + 3*6.0) / 5 = 4.2 is distinguishable from a plain batch mean of 3.75.
_BATCHES = [[_Datum([1.0, 1.0, 0.0])], [_Datum([1.0, 1.0, 1.0])]]
_OUTPUTS = [_Output([[-1.0, -2.0, -3.0]]), _Output([[-4.0, -6.0, -8.0]])]
_EXPECTED_NLL = 4.2

# A fully loss-masked batch: every token weight 0, so it carries no signal.
_MASKED_BATCH = [_Datum([0.0, 0.0])]


def _run_tinker(client, batches):
    return asyncio.run(TinkerSFTBackend._validate(client, _ValDataset(batches), compute_mean_nll))


def _run_fireworks(client, batches):
    return FireworksSFTBackend._validate(client, _ValDataset(batches))


_BACKENDS = [
    pytest.param(_FakeTinkerClient, _run_tinker, "test/mean_nll", id="tinker"),
    pytest.param(_FakeFireworksClient, _run_fireworks, "test/loss", id="fireworks"),
]


@pytest.mark.parametrize(("client_cls", "run_validate", "metric_key"), _BACKENDS)
def test_validate_never_runs_backward(client_cls, run_validate, metric_key):
    """Validation scores every batch through the forward-only API.

    A ``forward_backward`` here would land in the server's gradient accumulator
    and be applied by the next ``optim_step`` — the val set would be trained on.
    """
    client = client_cls(_OUTPUTS)
    metrics = run_validate(client, _BATCHES)
    assert client.calls == [("forward", "cross_entropy")] * len(_BATCHES)
    assert list(metrics) == [metric_key]


@pytest.mark.parametrize(("client_cls", "run_validate", "metric_key"), _BACKENDS)
def test_validate_returns_token_weighted_nll(client_cls, run_validate, metric_key):
    """The reported metric is the token-weighted mean NLL over the whole val set,
    reduced client-side from per-token logprobs on both backends."""
    metrics = run_validate(client_cls(_OUTPUTS), _BATCHES)
    assert metrics[metric_key] == pytest.approx(_EXPECTED_NLL)


@pytest.mark.parametrize(("client_cls", "run_validate", "metric_key"), _BACKENDS)
def test_validate_skips_fully_masked_batch(client_cls, run_validate, metric_key):
    """A batch with no loss tokens is not submitted and cannot poison the metric.

    Reachable whenever ``data.max_length`` truncates rows past their trainable
    tail — the dataset warns about it but still yields the datum.
    """
    batches = [_BATCHES[0], _MASKED_BATCH, _BATCHES[1]]
    client = client_cls(_OUTPUTS)
    metrics = run_validate(client, batches)
    assert metrics[metric_key] == pytest.approx(_EXPECTED_NLL)
    assert client.calls == [("forward", "cross_entropy")] * len(_BATCHES)


@pytest.mark.parametrize(("client_cls", "run_validate", "metric_key"), _BACKENDS)
def test_validate_rejects_an_entirely_masked_dataset(
    client_cls,
    run_validate,
    metric_key,
):
    del metric_key
    client = client_cls([])
    with pytest.raises(SFTConfigError, match="no trainable tokens"):
        run_validate(client, [_MASKED_BATCH])
    assert client.calls == []


class _EventFuture:
    def __init__(self, events, event, value):
        self.events = events
        self.event = event
        self.value = value

    async def result_async(self):
        self.events.append(self.event)
        return self.value


class _EventSyncFuture:
    def __init__(self, events, event, value):
        self.events = events
        self.event = event
        self.value = value

    def result(self, timeout=None):
        del timeout
        self.events.append(self.event)
        return self.value


def _output_for(data):
    return _Output([[-1.0] * datum.model_input.length for datum in data])


class _LoopDataset:
    def __init__(self, events, batches):
        self.events = events
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, seed):
        self.events.append(f"epoch-{seed}")

    def get_batch(self, index):
        self.events.append(f"batch-{index}")
        return self.batches[index]


class _LoopTinkerClient:
    def __init__(self, events):
        self.events = events
        self.last_label = None
        self.validation_count = 0

    async def forward_async(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        event = f"validate-{self.validation_count}"
        self.validation_count += 1
        self.events.append(event)
        return _EventFuture(self.events, f"finish-{event}", _output_for(data))

    async def forward_backward_async(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        self.last_label = data[0].label
        self.events.append(f"submit-fb-{self.last_label}")
        return _EventFuture(self.events, f"finish-fb-{self.last_label}", _output_for(data))

    async def optim_step_async(self, adam):
        del adam
        label = self.last_label
        self.events.append(f"submit-opt-{label}")
        return _EventFuture(self.events, f"finish-opt-{label}", SimpleNamespace())


class _LoopFireworksClient:
    def __init__(self, events):
        self.events = events
        self.last_label = None
        self.validation_count = 0

    def forward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        event = f"validate-{self.validation_count}"
        self.validation_count += 1
        self.events.append(event)
        return _output_for(data)

    def submit_forward_backward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        self.last_label = data[0].label
        self.events.append(f"submit-fb-{self.last_label}")
        output = SimpleNamespace(metrics={"response_tokens": 2, "loss:sum": 2.0})
        return _EventSyncFuture(self.events, f"finish-fb-{self.last_label}", output)

    def submit_optim_step(self, adam):
        del adam
        label = self.last_label
        self.events.append(f"submit-opt-{label}")
        return _EventSyncFuture(self.events, f"finish-opt-{label}", SimpleNamespace())


class _LoopTracking:
    def __init__(self, events, **kwargs):
        del kwargs
        self.events = events

    def log(self, data, step):
        self.events.append(f"log-{step}-{data.get('status', 'metrics')}")

    def finish(self):
        self.events.append("tracking-finish")


def _source_dataset():
    return Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ],
        name="validation-loop",
        split="train",
    )


def _assert_validation_boundary(events):
    assert events.index("validate-0") < events.index("batch-0")
    assert events.index("finish-opt-1") < events.index("validate-1")
    assert events.index("validate-1") < events.index("batch-2")
    assert [event for event in events if event.startswith("log-")] == [
        "log-0-metrics",
        "log-1-metrics",
        "log-2-metrics",
        "log-3-metrics",
        "log-3-completed",
    ]


def test_tinker_fit_drains_training_before_completed_step_validation(monkeypatch, tmp_path):
    """Step-2 validation sees both completed updates and no step-3 request."""
    from tinker_cookbook import checkpoint_utils, display

    import rllm.trainer.sft.tinker_backend as backend_module
    import rllm.utils.tracking as tracking_module

    events = []
    train_batches = [[_Datum([0.0, 1.0], label=i)] for i in range(3)]
    train_dataset = _LoopDataset(events, train_batches)
    val_dataset = _ValDataset([[_Datum([0.0, 1.0])]])
    client = _LoopTinkerClient(events)

    class _ServiceClient:
        def __init__(self, base_url=None):
            del base_url

        async def create_lora_training_client_async(self, **kwargs):
            del kwargs
            return client

    async def _save_checkpoint(**kwargs):
        events.append(f"save-{kwargs['name']}")

    source = _source_dataset()
    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=source,
            val_dataset=source,
            batch_size=1,
            output_dir=str(tmp_path),
            overrides={"trainer": {"total_epochs": 1, "save_freq": -1, "test_freq": 2}},
        )
    )
    backend.build_config()
    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (object(), train_dataset, val_dataset))
    monkeypatch.setattr(tinker, "ServiceClient", _ServiceClient)
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: None)
    monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", _save_checkpoint)
    monkeypatch.setattr(display, "colorize_example", lambda *_: "example")
    monkeypatch.setattr(tracking_module, "Tracking", lambda **kwargs: _LoopTracking(events, **kwargs))

    asyncio.run(backend._fit_async())

    _assert_validation_boundary(events)


def test_fireworks_fit_drains_training_before_completed_step_validation(monkeypatch, tmp_path):
    """The synchronous hosted loop observes the same validation boundary."""
    checkpoints_module = pytest.importorskip("training.utils.checkpoints")

    import rllm.trainer.sft.fireworks_backend as backend_module
    import rllm.utils.tracking as tracking_module

    events = []
    train_batches = [[_Datum([0.0, 1.0], label=i)] for i in range(3)]
    train_dataset = _LoopDataset(events, train_batches)
    val_dataset = _ValDataset([[_Datum([0.0, 1.0])]])
    client = _LoopFireworksClient(events)

    class _Checkpoints:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def resume(self):
            return None

        def save(self, name, **kwargs):
            del kwargs
            events.append(f"save-{name}")

        def promote_latest(self, output_model_id, base_model):
            del base_model
            return {"name": output_model_id}

    infra = SimpleNamespace(
        policy=client,
        service=object(),
        policy_job_id="fake-job",
        close=lambda: events.append("infra-close"),
    )
    source = _source_dataset()
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=source,
            val_dataset=source,
            batch_size=1,
            output_dir=str(tmp_path),
            overrides={"trainer": {"total_epochs": 1, "save_freq": -1, "test_freq": 2}},
        )
    )
    backend.build_config()
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (object(), train_dataset, val_dataset))
    monkeypatch.setattr(backend, "_provision", lambda config, api_key, base_url: infra)
    monkeypatch.setattr(checkpoints_module, "TrainingCheckpoints", _Checkpoints)
    monkeypatch.setattr(tracking_module, "Tracking", lambda **kwargs: _LoopTracking(events, **kwargs))

    backend.fit()

    _assert_validation_boundary(events)
