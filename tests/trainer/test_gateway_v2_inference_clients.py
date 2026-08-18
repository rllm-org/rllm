import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
from rllm_model_gateway.v2 import TokenInput

from rllm.trainer.fireworks.fireworks_inference_client import FireworksInferenceClient
from rllm.trainer.tinker.tinker_inference_client import TinkerInferenceClient
from rllm.trainer.verl.verl_inference_client import VerlInferenceClient


def _input(**sampling_params) -> TokenInput:
    return TokenInput(
        routing_key="session-1",
        prompt_token_ids=[10, 11],
        sampling_params=sampling_params,
    )


class FakeTinkerSamplingClient:
    def __init__(self, response) -> None:
        self.sample_async = AsyncMock(return_value=response)


def _tinker_client(response) -> TinkerInferenceClient:
    client = TinkerInferenceClient.__new__(TinkerInferenceClient)
    client._sampling_client = FakeTinkerSamplingClient(response)
    client._weight_version = 4
    client._max_prompt_length = 8
    client._max_response_length = 6
    client._max_model_length = 7
    return client


def test_tinker_translates_sampling_and_preserves_token_output() -> None:
    sequence = SimpleNamespace(tokens=[20, 21], logprobs=[-0.2, -0.3], stop_reason="stop")
    client = _tinker_client(SimpleNamespace(sequences=[sequence]))

    output = asyncio.run(
        client.generate(
            _input(
                max_tokens=10,
                temperature=0.7,
                stop_token_ids=[98, 99],
                stop=["ignored renderer stop"],
            )
        )
    )

    call = client._sampling_client.sample_async.await_args
    assert call.kwargs["num_samples"] == 1
    assert call.kwargs["prompt"].to_ints() == [10, 11]
    params = call.kwargs["sampling_params"]
    assert params.max_tokens == 5
    assert params.temperature == 0.7
    assert params.stop == [98, 99]
    assert output.completion_token_ids == [20, 21]
    assert output.logprobs == [-0.2, -0.3]
    assert output.finish_reason == "stop"
    assert output.weight_version == 4


def test_tinker_rejects_malformed_backend_output() -> None:
    sequence = SimpleNamespace(tokens=[20, 21], logprobs=[-0.2], stop_reason="stop")
    client = _tinker_client(SimpleNamespace(sequences=[sequence]))

    with pytest.raises(RuntimeError, match="different number of token IDs and logprobs"):
        asyncio.run(client.generate(_input()))


def test_tinker_update_replaces_client_and_weight_version() -> None:
    import pickle

    replacement = SimpleNamespace(name="replacement")
    client = _tinker_client(SimpleNamespace(sequences=[]))

    asyncio.run(client.update({"sampling_client": pickle.dumps(replacement), "weight_version": 9}))

    assert client._sampling_client.name == "replacement"
    assert client._weight_version == 9


@dataclass
class FakeFireworksMetrics:
    queue_time: float | None = 0.25
    inference_time: float | None = 0.5
    omitted: float | None = None


class FakeFireworksSampler:
    def __init__(self, *results) -> None:
        self.async_completions_stream = AsyncMock(side_effect=results)
        self.close = Mock()


def _fireworks_success():
    return (
        {
            "choices": [
                {
                    "raw_output": {"completion_token_ids": [20, 21]},
                    "logprobs": {
                        "content": [
                            {"logprob": -0.2, "routing_matrix": "matrix-1"},
                            {"logprob": -0.3, "routing_matrix": "matrix-2"},
                        ]
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        FakeFireworksMetrics(),
    )


def _fireworks_client(*results) -> FireworksInferenceClient:
    client = FireworksInferenceClient.__new__(FireworksInferenceClient)
    client._sampling_client = FakeFireworksSampler(*results)
    client._weight_version = 5
    client._max_prompt_length = 8
    client._max_response_length = 6
    client._max_model_length = 7
    client._sample_timeout = 30
    client._router_replay = True
    return client


def test_fireworks_translates_sampling_routing_and_output() -> None:
    client = _fireworks_client(_fireworks_success())

    output = asyncio.run(client.generate(_input(max_tokens=10, temperature=0.6, stop_token_ids=[98, 99])))

    call = client._sampling_client.async_completions_stream.await_args
    assert call.kwargs["prompt"] == [10, 11]
    assert call.kwargs["max_tokens"] == 5
    assert call.kwargs["temperature"] == 0.6
    assert call.kwargs["user"] == "session-1"
    assert call.kwargs["logprobs"] is True
    assert call.kwargs["include_routing_matrix"] is True
    assert "stop_token_ids" not in call.kwargs
    assert output.completion_token_ids == [20, 21]
    assert output.logprobs == [-0.2, -0.3]
    assert output.routed_experts == ["matrix-1", "matrix-2"]
    assert output.weight_version == 5
    assert output.metadata == {"queue_time": 0.25, "inference_time": 0.5}


def test_fireworks_retries_transient_failure(monkeypatch) -> None:
    request = httpx.Request("POST", "https://example.test/inference")
    client = _fireworks_client(httpx.ConnectError("connection failed", request=request), _fireworks_success())
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    output = asyncio.run(client.generate(_input()))

    assert output.completion_token_ids == [20, 21]
    assert client._sampling_client.async_completions_stream.await_count == 2
    sleep.assert_awaited_once_with(10)


def test_fireworks_update_and_close() -> None:
    client = _fireworks_client(_fireworks_success())

    asyncio.run(client.update({"weight_version": 8}))
    asyncio.run(client.close())

    assert client._weight_version == 8
    client._sampling_client.close.assert_called_once_with()


class FakeRayActorError(Exception):
    pass


class FakeRaySystemError(Exception):
    pass


class FakeVerlSamplingClient:
    def __init__(self, response) -> None:
        self.generate = AsyncMock(return_value=response)


def _verl_client(response) -> VerlInferenceClient:
    client = VerlInferenceClient.__new__(VerlInferenceClient)
    client._ray = SimpleNamespace(
        exceptions=SimpleNamespace(
            RayActorError=FakeRayActorError,
            RaySystemError=FakeRaySystemError,
        )
    )
    client._sampling_client = FakeVerlSamplingClient(response)
    client._weight_version = 6
    client._max_prompt_length = 8
    client._max_response_length = 4
    return client


def test_verl_translates_sampling_routing_and_router_replay() -> None:
    response = SimpleNamespace(
        token_ids=[20, 21],
        log_probs=[-0.2, -0.3],
        stop_reason="stop",
        extra_fields={"server": "metric"},
        routed_experts=[np.array([1]), np.array([2]), np.array([3])],
        num_preempted=2,
    )
    client = _verl_client(response)

    output = asyncio.run(client.generate(_input(max_tokens=2, temperature=0.5, stop_token_ids=[99])))

    call = client._sampling_client.generate.await_args
    assert call.kwargs["request_id"] == "session-1"
    assert call.kwargs["prompt_ids"] == [10, 11]
    assert call.kwargs["sampling_params"] == {
        "max_tokens": 2,
        "temperature": 0.5,
        "stop_token_ids": [99],
        "logprobs": 1,
    }
    assert output.completion_token_ids == [20, 21]
    assert output.logprobs == [-0.2, -0.3]
    assert len(output.routed_experts) == 2
    assert output.routed_experts[-1] == ""
    assert output.finish_reason == "length"
    assert output.weight_version == 6
    assert output.metadata == {"server": "metric", "num_preempted": 2}


def test_verl_rejects_missing_logprobs() -> None:
    response = SimpleNamespace(
        token_ids=[20],
        log_probs=None,
        stop_reason="stop",
        extra_fields={},
        routed_experts=None,
        num_preempted=None,
    )
    client = _verl_client(response)

    with pytest.raises(RuntimeError, match="no completion logprobs"):
        asyncio.run(client.generate(_input()))


def test_verl_update_changes_weight_version() -> None:
    client = _verl_client(None)

    asyncio.run(client.update({"weight_version": 10}))

    assert client._weight_version == 10
