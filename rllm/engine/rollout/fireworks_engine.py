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
import contextvars
import dataclasses
import logging
from typing import Any, cast

from fireworks.training.sdk import DeploymentSampler
from typing_extensions import override

from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.engine.rollout.tinker_engine import (
    TinkerEngine,
    _flat_token_input_length,
)
from rllm.engine.rollout.types import (
    TinkerTokenInput,
    TinkerTokenOutput,
    Tokenizer,
)
from rllm.types import TerminationEvent, TerminationReason

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


# Per-request inference headers (e.g. Fireworks session-affinity) injected into
# DeploymentSampler requests. ``DeploymentSampler.async_completions_stream`` only
# forwards body params and a fixed set of client-level headers, so there is no
# per-call header hook; we patch ``_inference_headers`` to merge whatever the
# current async context stashed here. A ContextVar is async-safe — each rollout
# task carries its own copy, so concurrent trajectories don't clobber each
# other's session-affinity key.
_per_request_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("rllm_fw_request_headers", default=None)


def _install_inference_header_patch() -> None:
    """Patch ``DeploymentSampler._inference_headers`` to merge per-request headers.

    Idempotent: re-imports / repeated calls are no-ops.
    """
    orig = DeploymentSampler._inference_headers
    if getattr(orig, "_rllm_session_affinity_patch", False):
        return

    def _inference_headers(self):  # noqa: ANN001 - matches SDK signature
        headers = orig(self)
        extra = _per_request_headers.get()
        if extra:
            headers = {**headers, **extra}
        return headers

    _inference_headers._rllm_session_affinity_patch = True  # type: ignore[attr-defined]
    DeploymentSampler._inference_headers = _inference_headers


_install_inference_header_patch()


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

    # Signals the gateway handler to forward a per-trajectory ``rllm_session_id``
    # so this engine can set Fireworks session-affinity headers (prefix-cache
    # reuse across a rollout's turns). Other engines ignore the session id.
    supports_session_affinity = True

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
        renderer_name: str | None = None,
        renderer_family: str = "auto",
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
            reasoning_effort: Reasoning effort for the chat parser and Fireworks
                completions API (e.g. ``low``, ``medium``, ``high``, ``none``).
            sample_timeout: HTTP timeout (seconds) for sampling calls.
            processor: Optional ``ProcessorMixin`` for multimodal models.
            router_replay: If True, request and propagate routing matrices
                for Router Replay (R3) training.
        """
        from rllm.engine.rollout.rollout_engine import RolloutEngine
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

        # Resolve the renderer the same way the gateway does, so the engine renders
        # turn 0 with the same renderer the gateway uses for the turn-1+ cumulative
        # bridge. resolve() auto-detects Fireworks-cookbook models (e.g. GLM-5.2 ->
        # "glm5") with no config. We adopt the unified renderer — and skip
        # ChatTemplateParser, whose eager apply_chat_template check rejects some
        # served templates (GLM-5.2 -> "'str object' has no attribute 'items'") —
        # when it is explicitly pinned, or when auto-detection lands on a
        # Fireworks-cookbook renderer (a model chat_parser can't serve). prime-rl
        # models under plain auto keep the existing chat_parser path (no regression).
        from rllm.renderers import resolve

        res = resolve(
            getattr(tokenizer, "name_or_path", None),
            tokenizer,
            backend="fireworks",
            family=renderer_family,
            renderer_name=renderer_name,
        )
        explicit = renderer_name is not None or renderer_family != "auto"
        self.renderer = res.renderer if (res.source == "tinker" or (explicit and res.source != "chat_template")) else None
        if self.renderer is not None:
            logger.info("FireworksEngine rendering via %s renderer (%s)", res.name, res.source)
        elif explicit and res.source == "chat_template":
            logger.warning(
                "renderer_family=%r / renderer_name=%r did not resolve a native renderer for %s; using ChatTemplateParser.",
                renderer_family,
                renderer_name,
                getattr(tokenizer, "name_or_path", None),
            )

        # bypass_render_with_parser reflects who owns rendering+parsing: True means
        # ChatTemplateParser (built below); False means the unified renderer does.
        # The flag was previously hardcoded True even with a renderer active, which
        # was a lie — and is why assemble_model_output asserted on a None chat_parser.
        self.bypass_render_with_parser = self.renderer is None
        self.chat_parser = None if self.renderer is not None else ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=disable_thinking)

        self.sample_timeout = sample_timeout
        self.router_replay = router_replay
        self.sampling_client = sampler

    @override
    def assemble_model_output(self, token_input, token_output) -> ModelOutput:
        """Assemble the ModelOutput, parsing the completion via the unified renderer.

        When a unified renderer is active, chat_parser is None, so the parent's
        chat_parser path can't run. Parse the completion through the renderer's
        ``parse_response`` (a ``ParsedResponse``) instead. Falls back to the
        inherited (chat_parser) path otherwise. This covers both turn 0
        (``get_model_response``) and turns 1+ (the gateway's token-prompt path),
        which both call ``assemble_model_output``.
        """
        if self.renderer is None:
            return super().assemble_model_output(token_input, token_output)

        sampled = cast(TinkerTokenOutput, token_output)
        response_tokens, logprobs = sampled.tokens, sampled.logprobs
        parsed = self.renderer.parse_response(response_tokens)

        prompt_ids: list[int] = []
        for elem in token_input:
            chunk_tokens = getattr(elem, "tokens", None)  # tinker.EncodedTextChunk -> ints
            if chunk_tokens is not None:
                prompt_ids.extend(chunk_tokens)
            else:
                prompt_ids.append(elem)

        return ModelOutput(
            text=self.tokenizer.decode(response_tokens, skip_special_tokens=True),
            content=parsed.content or "",
            reasoning=parsed.reasoning_content or "",
            tool_calls=parsed.tool_calls or [],
            prompt_ids=prompt_ids,
            completion_ids=response_tokens,
            logprobs=logprobs,
            prompt_length=_flat_token_input_length(token_input),
            completion_length=len(response_tokens),
            finish_reason=sampled.stop_reason,
        )

    # ------------------------------------------------------------------
    # Token-in / token-out override
    # ------------------------------------------------------------------

    @override
    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", None)

        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)

        if self.renderer is not None:
            token_input = self.renderer.render_ids(messages, tools=tools or None, add_generation_prompt=True)
        else:
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
        sampled_sequence = await self.get_token_output_from_token_input(
            token_input=token_input,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
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

        # Per-trajectory session id (gateway forwards it for affinity); it must
        # NOT leak into the request body, so pop it before building sampling params.
        session_id = kwargs.pop("rllm_session_id", None)

        for key in ("temperature", "top_p", "top_k", "user", "reasoning_effort"):
            if key in kwargs:
                sampling_params[key] = kwargs.pop(key)

        if "reasoning_effort" not in sampling_params and self.reasoning_effort is not None:
            sampling_params["reasoning_effort"] = self.reasoning_effort

        if self.router_replay:
            sampling_params["include_routing_matrix"] = True

        # Fireworks routes requests carrying the same session id to the same
        # replica, so its per-replica prompt-prefix KV is reused across a
        # trajectory's turns. Set once per turn from the trajectory id.
        session_headers = None
        if session_id:
            session_headers = {
                "x-multi-turn-session-id": str(session_id),
                "x-session-affinity": str(session_id),
            }

        raw, server_metrics = await self._completions_with_retry(
            prompt_ids,
            max_tokens,
            sampling_params,
            session_headers=session_headers,
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
        session_headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], dict | None]:
        """Call ``DeploymentSampler.async_completions_stream`` with transient-error retries.

        ``session_headers``, when given, are merged into the request's HTTP
        headers (via the ``_inference_headers`` patch) so Fireworks routes this
        request to the trajectory's pinned replica. Returns
        (response_dict, server_metrics_dict)."""

        for attempt in range(_MAX_SAMPLE_ATTEMPTS):
            try:
                token = _per_request_headers.set(session_headers) if session_headers else None
                try:
                    result, server_metrics = await self.sampling_client.async_completions_stream(
                        prompt=prompt_ids,
                        max_tokens=max_tokens,
                        raw_output=True,
                        logprobs=True,
                        http_timeout=self.sample_timeout,
                        **sampling_kwargs,
                    )
                finally:
                    if token is not None:
                        _per_request_headers.reset(token)
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
