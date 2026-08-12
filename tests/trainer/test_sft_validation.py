"""In-loop SFT validation on the tinker/fireworks family.

Both backends accumulate gradients server-side across ``forward_backward``
calls until the next ``optim_step``, so a validation pass that runs backward
silently trains on the held-out set. These tests pin the val loop to the
forward-only API, pin the metric it reports, and pin the ``fit`` call sites —
all on fakes, so no Tinker/Fireworks spend.

The two backends share the assertions but not the call protocol: tinker's
``_validate`` is async over ``forward_async`` futures, fireworks' is sync over
``ReconnectableClient.forward`` (which blocks internally).
"""

import pytest

tinker = pytest.importorskip("tinker")
from tinker_cookbook.supervised.common import compute_mean_nll  # noqa: E402

from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend  # noqa: E402
from rllm.trainer.sft.tinker_backend import TinkerSFTBackend  # noqa: E402


def _tensor(values):
    import torch

    return tinker.TensorData.from_torch(torch.tensor(values, dtype=torch.float32))


class _Datum:
    """Only the fields ``_validate`` reads off a tinker Datum."""

    def __init__(self, weights):
        self.loss_fn_inputs = {"weights": _tensor(weights)}


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
_MASKED_OUTPUT = _Output([[-9.0, -9.0]])


def _run_tinker(client, batches):
    import asyncio

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
    """A batch with no loss tokens scores nan and must not poison the run's metric.

    Reachable whenever ``data.max_length`` truncates rows past their trainable
    tail — the dataset warns about it but still yields the datum.
    """
    batches = [_BATCHES[0], _MASKED_BATCH, _BATCHES[1]]
    outputs = [_OUTPUTS[0], _MASKED_OUTPUT, _OUTPUTS[1]]
    metrics = run_validate(client_cls(outputs), batches)
    assert metrics[metric_key] == pytest.approx(_EXPECTED_NLL)


@pytest.mark.parametrize(("client_cls", "run_validate", "metric_key"), _BACKENDS)
def test_validate_rejects_an_entirely_masked_dataset(
    client_cls,
    run_validate,
    metric_key,
):
    del metric_key
    with pytest.raises(SFTConfigError, match="no trainable tokens"):
        run_validate(client_cls([_MASKED_OUTPUT]), [_MASKED_BATCH])
