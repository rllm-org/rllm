"""RemoteRolloutClient -- drop-in replacement for ``RolloutClient`` that talks
to the remote TrainingServer API gateway instead of a local SGLang router.

The public interface (``generate``, ``chat_completion``) is intentionally kept
identical to :class:`rllm.experimental.fully_async.client.RolloutClient` so
that existing ``rollout_fn`` implementations work without modification.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from rllm.experimental.fully_async.protocol import OutputChunk, OutputWithVersion

logger = logging.getLogger(__name__)

# Retry parameters for 503 (server syncing weights) responses.
_INITIAL_BACKOFF = 0.25  # seconds
_MAX_BACKOFF = 8.0  # seconds
_BACKOFF_FACTOR = 2.0


class RemoteRolloutClient:
    """HTTP client that connects to a remote ``TrainingServer`` API gateway.

    The gateway proxies ``/v1/generate`` to the underlying SGLang router and
    manages weight-sync pauses transparently.  When the server is syncing
    (HTTP 503), this client automatically retries with exponential back-off.

    Parameters
    ----------
    server_url:
        Base URL of the remote TrainingServer (e.g. ``https://my-app.modal.run``).
    tokenizer:
        HuggingFace tokenizer – required for ``chat_completion``.
    max_concurrency:
        Maximum number of concurrent HTTP connections.
    max_tokens:
        Default context budget (prompt + response).
    """

    def __init__(
        self,
        server_url: str,
        tokenizer=None,
        max_concurrency: int = 4096,
        max_tokens: int = 32768,
    ):
        # Normalise URL (strip trailing slash)
        self.server_url = server_url.rstrip("/")
        self.tokenizer = tokenizer
        self._max_concurrency = max_concurrency

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._max_concurrency,
                max_keepalive_connections=min(self._max_concurrency, 1000),
            ),
            timeout=httpx.Timeout(None),
        )

        self.cur_version: int = 0
        self.max_tokens: int = max_tokens

        # Lazy-initialised tool parser (only needed for chat_completion)
        self._parser = None

    # ------------------------------------------------------------------
    # Properties kept for compatibility with RolloutClient
    # ------------------------------------------------------------------

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def set_version(self, version: int):
        self.cur_version = version

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _post_with_retry(self, url: str, payload: dict) -> httpx.Response:
        """POST *payload* to *url*, retrying on 503 (server syncing)."""
        backoff = _INITIAL_BACKOFF
        while True:
            response = await self.client.post(url, json=payload)
            if response.status_code == 503:
                logger.debug("Server syncing weights – retrying in %.2fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_FACTOR, _MAX_BACKOFF)
                continue
            response.raise_for_status()
            # Update param version from response header if provided
            hdr_version = response.headers.get("X-Param-Version")
            if hdr_version is not None:
                self.cur_version = int(hdr_version)
            return response

    # ------------------------------------------------------------------
    # Low-Level API – token-level generation
    # ------------------------------------------------------------------

    async def generate(self, prompt_ids: list[int], sampling_params: dict) -> OutputWithVersion:
        """Generate with token IDs directly (low-level API).

        Mirrors :meth:`RolloutClient.generate`.
        """
        output = OutputWithVersion(prompt_ids=prompt_ids, output_chunks=[])

        while True:
            output, sampling_params = await self._generate(output, sampling_params)
            if output.finish_reason == "abort":
                # Request was aborted during weight sync – retry seamlessly
                continue
            return output

    async def _generate(self, output: OutputWithVersion, sampling_params: dict):
        """Single request/response cycle through the API gateway."""
        old_version = self.cur_version

        payload = {
            "input_ids": output.all_tokens(),
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        resp = await self._post_with_retry(self.server_url + "/v1/generate", payload)
        data = resp.json()

        # Parse finish reason
        finish_reason_obj = data["meta_info"].get("finish_reason")
        output.finish_reason = finish_reason_obj["type"] if finish_reason_obj else "unknown"

        # Parse logprobs
        output_token_logprobs = data["meta_info"].get("output_token_logprobs", [])
        logprob_values = [float(lp) for lp, _tid, _ in output_token_logprobs]

        chunk = OutputChunk(
            response_ids=data["output_ids"],
            response_logprobs=logprob_values,
            version=old_version if output.finish_reason == "abort" else self.cur_version,
        )
        output.append(chunk)

        # Adjust max_tokens for continuation after partial generation
        max_tok = sampling_params.get("max_new_tokens") or sampling_params.get("max_tokens")
        if max_tok is None:
            return output, sampling_params

        sampling_params = sampling_params.copy()
        remaining = max_tok - len(chunk.response_ids)
        if "max_new_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = remaining
        else:
            sampling_params["max_tokens"] = remaining

        return output, sampling_params

    # ------------------------------------------------------------------
    # High-Level Chat API
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], OutputWithVersion]:
        """Chat completion – same interface as :meth:`RolloutClient.chat_completion`."""
        from rllm.experimental.fully_async.message_utils import parse_response
        from rllm.parser.tool_parser import ToolParser

        if self.tokenizer is None:
            raise ValueError("tokenizer is required for chat_completion")

        if self._parser is None:
            self._parser = ToolParser.get_parser(self.tokenizer)

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)

        sampling_params = sampling_params or {}
        if sampling_params.get("max_new_tokens") is None:
            sampling_params["max_new_tokens"] = self.max_tokens - len(prompt_ids)

        output = await self.generate(prompt_ids, sampling_params)
        message = parse_response(self.tokenizer, self._parser, output.all_response_ids())
        return message, output

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        """Fetch training status from the server."""
        resp = await self.client.get(self.server_url + "/v1/status")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self.client.aclose()
