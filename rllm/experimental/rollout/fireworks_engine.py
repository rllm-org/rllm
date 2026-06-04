"""RolloutEngine backed by Fireworks ``DeploymentSampler``.

Inherits from ``TinkerEngine``. The only differences are:

1. ``__init__``: creates a ``DeploymentSampler`` instead of requiring a
   ``tinker.ServiceClient``.  The sampler is stored as ``self.sampling_client``
   so that the inherited ``set_sampling_client`` / ``generate_episodes`` flow
   works unchanged.
2. ``get_token_output_from_token_input``: calls ``DeploymentSampler.completions``
   (token-in / token-out) and wraps the response in a ``SampledSequence``-compatible
   adapter so that the inherited ``assemble_model_output`` works unchanged.

Everything else, including ``get_model_response``, ``assemble_model_output``,
``set_sampling_client``, ``_prepare_max_tokens``, and chat-template rendering,
is inherited from ``TinkerEngine``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any

from fireworks.training.sdk import DeploymentSampler
from typing_extensions import override

from rllm.experimental.rollout.rollout_engine import ModelOutput
from rllm.experimental.rollout.tinker_engine import (
    TinkerEngine,
    _flat_token_input_length,
)
from rllm.experimental.rollout.types import (
    TinkerTokenInput,
    TinkerTokenOutput,
    Tokenizer,
)
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)

_MAX_SAMPLE_ATTEMPTS = 5
_TRANSIENT_ERROR_MARKERS = (
    "502",
    "503",
    "425",
    "Connection",
    "incomplete chunked read",
    "_SSETruncationError",
    "closed the SSE stream mid-generation",
)


class _EmptyCompletionIdsError(RuntimeError):
    pass


class _SampledSequenceAdapter:
    """Lightweight adapter so that a ``DeploymentSampler.completions`` response
    exposes the same ``.tokens``, ``.logprobs``, ``.stop_reason`` interface
    that ``tinker.SampledSequence`` (``TinkerTokenOutput``) provides."""

    __slots__ = ("tokens", "logprobs", "stop_reason", "routing_matrices", "server_metrics")

    def __init__(
        self,
        tokens: list[int],
        logprobs: list[float] | None,
        stop_reason: str | None,
        routing_matrices: list[str] | None = None,
        server_metrics: dict | None = None,
    ):
        self.tokens = tokens
        self.logprobs = logprobs
        self.stop_reason = stop_reason
        self.routing_matrices = routing_matrices
        self.server_metrics = server_metrics


class FireworksEngine(TinkerEngine):
    """``TinkerEngine`` subclass that uses a Fireworks ``DeploymentSampler``
    for inference instead of a Tinker ``SamplingClient``.

    ``DeploymentSampler`` supports token-in / token-out via the
    ``/inference/v1/completions`` endpoint, so ``TinkerTokenInput`` and
    ``TinkerTokenOutput`` are fully supported.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        sampler: DeploymentSampler,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        max_model_length: int = 32768,
        sampling_params: dict | None = None,
        disable_thinking: bool = False,
        accumulate_reasoning: bool = False,
        reasoning_effort: str = "medium",
        sample_timeout: int = 600,
        processor=None,
        router_replay: bool = False,
        **kwargs,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer for chat-template rendering.
            sampler: Pre-built ``DeploymentSampler``.
            max_prompt_length: Hard cap on prompt token length.
            max_response_length: Default max completion tokens.
            max_model_length: Total context window.
            sampling_params: Dict with optional ``"train"`` / ``"val"``
                sub-dicts for default sampling kwargs.
            disable_thinking: Suppress thinking tokens in the prompt.
            accumulate_reasoning: Accumulate reasoning across turns.
            reasoning_effort: Reasoning effort hint for the parser.
            sample_timeout: HTTP timeout (seconds) for sampling calls.
            processor: Optional ``ProcessorMixin`` for multimodal models.
            router_replay: If True, request and propagate routing matrices
                for Router Replay (R3) training.
        """
        from rllm.experimental.rollout.rollout_engine import RolloutEngine
        from rllm.parser import ChatTemplateParser

        # Skip TinkerEngine.__init__ (it requires tinker.ServiceClient);
        # set up the same attributes directly.
        RolloutEngine.__init__(self)

        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_model_length - 1 if max_model_length is not None else max_prompt_length + max_response_length - 1
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = reasoning_effort

        self.train_sampling_params = dict((sampling_params or {}).get("train", {}))
        self.val_sampling_params = dict((sampling_params or {}).get("val", {}))

        # Chat template parser (same setup as TinkerEngine bypass mode)
        self.bypass_render_with_parser = True
        self.chat_parser = ChatTemplateParser.get_parser(
            tokenizer,
            processor=processor,
            disable_thinking=disable_thinking,
        )

        self.sample_timeout = sample_timeout
        self.router_replay = router_replay
        self.sampling_client = sampler

    # ------------------------------------------------------------------
    # Token-in / token-out override
    # ------------------------------------------------------------------

    @override
    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", None)

        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)

        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            reasoning_effort=reasoning_effort,
            accumulate_reasoning=accumulate_reasoning,
        )
        token_input = self.tokenizer.encode(prompt, add_special_tokens=False)

        if application_id is not None:
            kwargs["user"] = application_id

        version = self.weight_version
        sampled_sequence = await self.get_token_output_from_token_input(token_input=token_input, **kwargs)
        result = self.assemble_model_output(token_input=token_input, token_output=sampled_sequence)
        result.weight_version = version
        result.routing_matrices = sampled_sequence.routing_matrices
        result.metrics = sampled_sequence.server_metrics
        return result

    @override
    async def get_model_response_from_tokens(self, token_input, **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", None)
        if application_id is not None:
            kwargs["user"] = application_id

        version = self.weight_version
        sampled_sequence = await self.get_token_output_from_token_input(token_input=token_input, **kwargs)
        result = self.assemble_model_output(token_input=token_input, token_output=sampled_sequence)
        result.weight_version = version
        result.routing_matrices = sampled_sequence.routing_matrices
        result.metrics = sampled_sequence.server_metrics
        return result

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    async def compute_logprobs(self, ids: list[int]) -> list[float]:
        raise NotImplementedError("compute_logprobs is not supported by FireworksEngine.")

    @override
    async def get_token_output_from_token_input(self, token_input: TinkerTokenInput, **kwargs) -> TinkerTokenOutput:
        """Sample from the Fireworks deployment using pre-tokenized IDs.

        Returns a ``SampledSequence``-compatible object so that the inherited
        ``assemble_model_output`` works unchanged.
        """
        if self.sampling_client is None:
            raise RuntimeError("Sampling client not set. Call set_sampling_client() first.")

        input_length = _flat_token_input_length(token_input)

        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        if enforce_max_prompt_length and (input_length > self.max_prompt_length or input_length >= self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # Flatten TinkerTokenInput to plain list[int]
        prompt_ids: list[int] = []
        for elem in token_input:
            if isinstance(elem, int):
                prompt_ids.append(elem)
            else:
                # tinker.EncodedTextChunk
                prompt_ids.extend(elem.tokens)

        sampling_params = self.val_sampling_params.copy() if self.is_validation else self.train_sampling_params.copy()
        requested_max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self.max_response_length))
        requested_max_tokens = sampling_params.pop("max_tokens", requested_max_tokens)
        max_tokens = self._prepare_max_tokens(requested_max_tokens, input_length)

        for key in ("temperature", "top_p", "top_k", "user"):
            if key in kwargs:
                sampling_params[key] = kwargs.pop(key)

        if self.router_replay:
            sampling_params["include_routing_matrix"] = True

        raw, server_metrics = await self._completions_with_retry(
            prompt_ids,
            max_tokens,
            sampling_params,
        )

        choice = raw["choices"][0]
        completion_ids: list[int] = list((choice.get("raw_output") or {}).get("completion_token_ids") or [])

        logprobs: list[float] | None = None
        content: list[dict] | None = None
        lp_data = choice.get("logprobs")
        if lp_data and isinstance(lp_data, dict):
            content = lp_data.get("content")
            if isinstance(content, list) and content:
                logprobs = [tok.get("logprob", 0.0) for tok in content]

        finish_reason = choice.get("finish_reason", "stop")

        routing_matrices = None
        if self.router_replay and content:
            matrices = [tok.get("routing_matrix", "") for tok in content]
            if any(matrices):
                routing_matrices = matrices
            else:
                logger.debug("router_replay enabled but API returned no routing matrices")

        if logprobs is not None and len(logprobs) != len(completion_ids):
            raise RuntimeError(f"Fireworks response length mismatch: {len(logprobs)} logprobs vs {len(completion_ids)} completion tokens")
        if routing_matrices is not None and len(routing_matrices) != len(completion_ids):
            raise RuntimeError(f"Fireworks response length mismatch: {len(routing_matrices)} routing matrices vs {len(completion_ids)} completion tokens")

        return _SampledSequenceAdapter(  # type: ignore[return-value]
            tokens=completion_ids,
            logprobs=logprobs,
            stop_reason=finish_reason,
            routing_matrices=routing_matrices,
            server_metrics=server_metrics,
        )

    # ------------------------------------------------------------------
    # Internal retry helper
    # ------------------------------------------------------------------

    async def _completions_with_retry(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        sampling_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict | None]:
        """Call ``DeploymentSampler.async_completions_stream`` with transient-error retries.

        Returns (response_dict, server_metrics_dict)."""

        for attempt in range(_MAX_SAMPLE_ATTEMPTS):
            try:
                result, server_metrics = await self.sampling_client.async_completions_stream(
                    prompt=prompt_ids,
                    max_tokens=max_tokens,
                    raw_output=True,
                    logprobs=True,
                    http_timeout=self.sample_timeout,
                    **sampling_kwargs,
                )
                metrics_dict = {k: v for k, v in dataclasses.asdict(server_metrics).items() if v is not None} if server_metrics else None
                choice = (result.get("choices") or [{}])[0]
                completion_ids = (choice.get("raw_output") or {}).get("completion_token_ids") or []
                if not completion_ids:
                    raise _EmptyCompletionIdsError("Fireworks response included empty completion_token_ids")
                return result, metrics_dict
            except Exception as exc:
                err = str(exc)
                exc_name = exc.__class__.__name__
                transient = isinstance(exc, _EmptyCompletionIdsError) or any(marker in err or marker in exc_name for marker in _TRANSIENT_ERROR_MARKERS)
                if transient and attempt < _MAX_SAMPLE_ATTEMPTS - 1:
                    wait = 10 * (attempt + 1)
                    logger.debug(
                        "Attempt %d/%d failed (%s), retrying in %ds...",
                        attempt + 1,
                        _MAX_SAMPLE_ATTEMPTS,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp_text = getattr(getattr(exc, "response", None), "text", None)
                logger.error(
                    "Sampling failed permanently after %d attempts: %s\n%s",
                    attempt + 1,
                    exc,
                    resp_text or "",
                )
                raise
        raise RuntimeError("unreachable")
