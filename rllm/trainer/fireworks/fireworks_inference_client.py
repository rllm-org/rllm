import asyncio
import dataclasses
import logging
import ssl
import time
from typing import Any

import httpx

from rllm_model_gateway.v2 import GatewayError, TokenInput, TokenOutput

_MAX_ATTEMPTS = 5
_RETRY_BUDGET_SECONDS = 90.0
_RETRY_BACKOFF_CAP_SECONDS = 15.0
_TRANSIENT_ERRORS = (
    "empty completion token IDs",
    "502",
    "503",
    "425",
    "429",
    "Connection",
    "incomplete chunked read",
    "_SSETruncationError",
    "closed the SSE stream mid-generation",
)

logger = logging.getLogger(__name__)


class FireworksInferenceClient:
    def __init__(
        self,
        sampling_client_kwargs: dict[str, Any],
        weight_version: int,
        max_prompt_length: int,
        max_response_length: int,
        max_model_length: int,
        sample_timeout: float,
        router_replay: bool,
    ) -> None:
        from fireworks.training.sdk import DeploymentSampler

        self._sampling_client = DeploymentSampler(
            tokenizer=None,
            **sampling_client_kwargs,
        )
        self._weight_version = weight_version
        self._max_prompt_length = max_prompt_length
        self._max_response_length = max_response_length
        self._max_model_length = max_model_length
        self._sample_timeout = sample_timeout
        self._router_replay = router_replay

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
        sampling_params.pop("stop_token_ids", None)
        sampling_params["logprobs"] = True
        sampling_params["user"] = request.session_id
        sampling_params["include_routing_matrix"] = self._router_replay

        started_at = time.monotonic()
        first_failure_at: float | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                raw, server_metrics = await self._sampling_client.async_completions_stream(
                    prompt=request.prompt_token_ids,
                    max_tokens=max_tokens,
                    raw_output=True,
                    http_timeout=self._sample_timeout,
                    **sampling_params,
                )
                choice = (raw.get("choices") or [{}])[0]
                completion_ids = list((choice.get("raw_output") or {}).get("completion_token_ids") or [])
                if not completion_ids:
                    raise RuntimeError("Fireworks response included empty completion token IDs")
                metrics = (
                    {key: value for key, value in dataclasses.asdict(server_metrics).items() if value is not None}
                    if server_metrics
                    else {}
                )
                break
            except Exception as exc:
                error = str(exc)
                error_type = exc.__class__.__name__
                transient = (
                    isinstance(exc, httpx.TimeoutException | httpx.TransportError | ssl.SSLError)
                    or any(marker in error or marker in error_type for marker in _TRANSIENT_ERRORS)
                )
                now = time.monotonic()
                if first_failure_at is None:
                    first_failure_at = now
                wait = min(10 * (attempt + 1), _RETRY_BACKOFF_CAP_SECONDS)
                within_budget = now - first_failure_at + wait < _RETRY_BUDGET_SECONDS
                if transient and attempt < _MAX_ATTEMPTS - 1 and within_budget:
                    logger.debug(
                        "Fireworks generation attempt %d/%d failed after %.1fs (%s: %s); retrying in %.1fs",
                        attempt + 1,
                        _MAX_ATTEMPTS,
                        now - started_at,
                        error_type,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                response = getattr(exc, "response", None)
                if not transient:
                    reason = "permanent failure"
                elif not within_budget:
                    reason = "retry budget exhausted"
                else:
                    reason = "retry attempts exhausted"
                logger.error(
                    "Fireworks generation failed (%s) after %d attempts / %.1fs (%s): %s; response=%s; headers=%s",
                    reason,
                    attempt + 1,
                    now - started_at,
                    error_type,
                    exc,
                    getattr(response, "text", None),
                    dict(getattr(response, "headers", None) or {}),
                )
                status_code = getattr(response, "status_code", None)
                if status_code in (400, 422):
                    raise GatewayError(str(exc), 400, "invalid_request_error") from exc
                if status_code == 429:
                    raise GatewayError(str(exc), 429, "rate_limit_error") from exc
                if status_code == 408:
                    raise GatewayError("Fireworks generation timed out", 504, "timeout_error") from exc
                if isinstance(exc, httpx.TimeoutException):
                    raise GatewayError("Fireworks generation timed out", 504, "timeout_error") from exc
                if isinstance(exc, httpx.TransportError | ssl.SSLError):
                    raise GatewayError("Fireworks inference service is unavailable", 503, "server_error") from exc
                raise
        else:
            raise RuntimeError("Fireworks generation retries exhausted")

        logprobs = None
        routed_experts = None
        content = (choice.get("logprobs") or {}).get("content")
        if content:
            logprobs = [float(token["logprob"]) for token in content]
            matrices = [token.get("routing_matrix", "") for token in content]
            if any(matrices):
                routed_experts = matrices
        if logprobs is not None and len(logprobs) != len(completion_ids):
            raise RuntimeError("Fireworks returned a different number of token IDs and logprobs")
        if routed_experts is not None and len(routed_experts) != len(completion_ids):
            raise RuntimeError("Fireworks returned a different number of token IDs and routing matrices")
        return TokenOutput(
            completion_token_ids=completion_ids,
            logprobs=logprobs,
            routed_experts=routed_experts,
            finish_reason=choice.get("finish_reason", "stop"),
            weight_version=weight_version,
            metadata=metrics,
        )

    async def update(self, update: dict[str, Any]) -> None:
        self._weight_version = update["weight_version"]

    async def close(self) -> None:
        self._sampling_client.close()
