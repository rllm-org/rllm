import base64
import io
from typing import Any

import numpy as np

from rllm_model_gateway.v2 import GatewayError, TokenInput, TokenOutput


class VerlInferenceClient:
    def __init__(self, sampling_client: bytes, weight_version: int, max_prompt_length: int, max_response_length: int) -> None:
        import ray

        self._sampling_client = ray.cloudpickle.loads(sampling_client)
        self._weight_version = weight_version
        self._max_prompt_length = max_prompt_length
        self._max_response_length = max_response_length

    async def generate(self, request: TokenInput) -> TokenOutput:
        if len(request.prompt_token_ids) > self._max_prompt_length:
            raise GatewayError("prompt exceeds the maximum prompt length")
        weight_version = self._weight_version
        sampling_params = dict(request.sampling_params)
        max_tokens = sampling_params.pop(
            "max_tokens",
            sampling_params.pop("max_completion_tokens", sampling_params.pop("max_new_tokens", self._max_response_length)),
        )
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise GatewayError("max_tokens must be a positive integer")
        sampling_params["max_tokens"] = max_tokens
        raw = await self._sampling_client.generate(
            request_id=request.session_id,
            prompt_ids=request.prompt_token_ids,
            sampling_params=sampling_params,
        )
        if raw.stop_reason in ("abort", "aborted"):
            raise RuntimeError("Verl rollout was aborted")
        if not raw.token_ids:
            raise RuntimeError("Verl returned no completion tokens")
        if raw.log_probs is not None and len(raw.log_probs) != len(raw.token_ids):
            raise RuntimeError("Verl returned a different number of token IDs and logprobs")
        metadata = dict(raw.extra_fields)
        routed_experts = None
        if raw.routed_experts is not None:
            routed = raw.routed_experts
            completion_length = len(raw.token_ids)
            expected_length = len(request.prompt_token_ids) + completion_length - 1
            if len(routed) != expected_length:
                raise RuntimeError(f"Verl returned {len(routed)} routed-expert rows, expected {expected_length}")
            routed_experts = []
            for row in routed[len(request.prompt_token_ids) :]:
                buffer = io.BytesIO()
                np.save(buffer, row)
                routed_experts.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            routed_experts.append("")
        finish_reason = "length" if len(raw.token_ids) >= max_tokens else "stop"
        if raw.num_preempted is not None:
            metadata["num_preempted"] = raw.num_preempted
        return TokenOutput(
            completion_token_ids=list(raw.token_ids),
            logprobs=list(raw.log_probs) if raw.log_probs is not None else None,
            routed_experts=routed_experts,
            finish_reason=finish_reason,
            weight_version=weight_version,
            metadata=metadata,
        )

    async def update(self, update: dict[str, Any]) -> None:
        self._weight_version = update["weight_version"]

    async def close(self) -> None:
        pass
