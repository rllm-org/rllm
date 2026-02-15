"""
Remote Rollout Engine for Remote Agent Support.

A ``RolloutEngine`` implementation that calls the trainer's inference API
endpoint ``POST /v1/model_response`` and returns the full ``ModelOutput``
(including prompt_ids, completion_ids, logprobs, etc.).

Unlike using ``OpenAIEngine`` against the trainer's ``/v1/chat/completions``
endpoint, this preserves all token-level information needed for RL training
(policy gradient, advantage computation, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from rllm.experimental.rollout.rollout_engine import ModelOutput, RolloutEngine

logger = logging.getLogger(__name__)


class RemoteRolloutEngine(RolloutEngine):
    """RolloutEngine that delegates inference to a remote inference API server.

    Calls ``POST <inference_api_url>/model_response`` which returns the full
    ``ModelOutput.to_dict()``, then deserializes it via ``ModelOutput.from_dict()``.

    Args:
        inference_api_url: Base URL of the trainer's inference API
            (e.g. ``http://trainer-host:8089/v1``).
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retry attempts on failure.
    """

    def __init__(
        self,
        inference_api_url: str,
        timeout: float = 300.0,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inference_api_url = inference_api_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Send messages to the remote inference API and return the full ModelOutput.

        Args:
            messages: List of chat message dicts (``{"role": ..., "content": ...}``).
            **kwargs: Sampling parameters (temperature, top_p, max_tokens, etc.).

        Returns:
            The complete ``ModelOutput`` including token IDs and logprobs.
        """
        # Build request payload matching ModelResponseRequest schema
        payload: dict[str, Any] = {
            "messages": messages,
        }

        # Map known kwargs to top-level fields
        for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "application_id"):
            if key in kwargs:
                payload[key] = kwargs.pop(key)

        # Remaining kwargs go into extra_params
        if kwargs:
            payload["extra_params"] = kwargs

        url = f"{self.inference_api_url}/model_response"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return ModelOutput.from_dict(data["model_output"])
            except Exception as e:
                last_error = e
                logger.warning(f"RemoteRolloutEngine call failed (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2**attempt, 10))

        raise RuntimeError(f"RemoteRolloutEngine failed after {self.max_retries} attempts: {last_error}") from last_error

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()
