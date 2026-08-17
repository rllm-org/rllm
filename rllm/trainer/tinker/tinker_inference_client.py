import pickle
from typing import Any

import tinker
from rllm_model_gateway.v2 import GatewayError, TokenInput, TokenOutput


class TinkerInferenceClient:
    def __init__(
        self,
        sampling_client: bytes,
        weight_version: int,
        max_prompt_length: int,
        max_response_length: int,
        max_model_length: int,
    ) -> None:
        self._sampling_client: tinker.SamplingClient = pickle.loads(sampling_client)
        self._weight_version = weight_version
        self._max_prompt_length = max_prompt_length
        self._max_response_length = max_response_length
        self._max_model_length = max_model_length

    async def generate(self, request: TokenInput) -> TokenOutput:
        if len(request.prompt_token_ids) > self._max_prompt_length:
            raise GatewayError("prompt exceeds the maximum prompt length")
        if len(request.prompt_token_ids) >= self._max_model_length:
            raise GatewayError("prompt exceeds the model context length")
        weight_version = self._weight_version
        sampling_params = dict(request.sampling_params)
        max_tokens = sampling_params.pop(
            "max_tokens",
            sampling_params.pop("max_completion_tokens", sampling_params.pop("max_new_tokens", self._max_response_length)),
        )
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise GatewayError("max_tokens must be a positive integer")
        max_tokens = min(max_tokens, self._max_model_length - len(request.prompt_token_ids))
        if max_tokens <= 0:
            raise GatewayError("prompt exceeds the model context length")
        stop = sampling_params.pop("stop_token_ids", None)
        sampling_params.pop("stop", None)
        sampling_client = self._sampling_client
        try:
            params = tinker.types.SamplingParams(max_tokens=max_tokens, stop=stop, **sampling_params)
        except (TypeError, ValueError) as exc:
            raise GatewayError(f"invalid sampling parameters: {exc}") from exc
        try:
            response = await sampling_client.sample_async(
                prompt=tinker.types.ModelInput.from_ints(request.prompt_token_ids),
                num_samples=1,
                sampling_params=params,
            )
        except (tinker.BadRequestError, tinker.UnprocessableEntityError) as exc:
            raise GatewayError(str(exc), 400, "invalid_request_error") from exc
        except tinker.RateLimitError as exc:
            raise GatewayError(str(exc), 429, "rate_limit_error") from exc
        except tinker.APITimeoutError as exc:
            raise GatewayError("Tinker generation timed out", 504, "timeout_error") from exc
        except tinker.APIConnectionError as exc:
            raise GatewayError("Tinker inference service is unavailable", 503, "server_error") from exc
        except (tinker.InternalServerError, tinker.SidecarError) as exc:
            raise GatewayError("Tinker inference service is unavailable", 503, "server_error") from exc
        if not response.sequences:
            raise RuntimeError("Tinker returned no sequences")
        sequence = response.sequences[0]
        if not sequence.tokens:
            raise RuntimeError("Tinker returned no completion tokens")
        if sequence.logprobs is not None and len(sequence.logprobs) != len(sequence.tokens):
            raise RuntimeError("Tinker returned a different number of token IDs and logprobs")
        return TokenOutput(
            completion_token_ids=sequence.tokens,
            logprobs=sequence.logprobs,
            finish_reason=sequence.stop_reason,
            weight_version=weight_version,
        )

    async def update(self, update: dict[str, Any]) -> None:
        self._sampling_client = pickle.loads(update["sampling_client"])
        self._weight_version = update["weight_version"]

    async def close(self) -> None:
        pass
