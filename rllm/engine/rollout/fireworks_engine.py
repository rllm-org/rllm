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
import functools
import logging
import ssl
import time
from typing import Any

import httpx
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
# The gateway wires FireworksEngine as an in-process handler, so these retries
# run *inside* the agent's HTTP call and hold that (byte-silent, no-heartbeat on
# the cumulative path) connection open the whole time. Retrying long enough to
# ride out a transient reset / weight-sync reload is good; holding past the
# client/tunnel tolerance is not — the client re-sends, surfacing as
# TokenAccumulator "duplicate" churn + wasted regeneration. So cap the total
# retry wall-clock: ride out the common case, then fail fast so a persistent
# outage surfaces instead of stalling the connection.
_RETRY_BUDGET_S = 90.0
# Per-retry backoff cap: the old 10/20/30/40s schedule alone could sleep ~100s,
# well past the budget. Cap it so backoff can't dominate the budget.
_RETRY_BACKOFF_CAP_S = 15.0
_TRANSIENT_ERROR_MARKERS = (
    "502",
    "503",
    "425",
    "429",
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


def _install_httpx_orjson_patch() -> None:
    """Serialize httpx ``json=`` request bodies with orjson instead of stdlib json.

    ``DeploymentSampler`` POSTs the full (≤120K-token) prompt via
    ``client.post(json=payload)``; httpx serializes it with stdlib ``json`` on the
    calling thread — the gateway's single event loop — which at high concurrency is
    the dominant per-request on-loop CPU cost. orjson is ~3x faster and matches
    httpx's wire semantics (compact, UTF-8, rejects NaN).

    Best-effort and guarded: httpx internals (``encode_json`` returning
    ``(headers, ByteStream)``) are version-specific, so we probe the shape against
    the real function before installing and skip (keeping stdlib) on any mismatch —
    a future httpx upgrade can degrade the speedup but never break sampling.
    """
    try:
        import httpx._content as _hc
        import orjson
        from httpx._content import ByteStream
    except Exception:  # noqa: BLE001 - httpx/orjson layout unknown → skip patch
        return

    orig = getattr(_hc, "encode_json", None)
    if orig is None or getattr(orig, "_rllm_orjson_patch", False):
        return

    def _encode_json(json):  # noqa: ANN001 - matches httpx signature
        try:
            body = orjson.dumps(json)
        except TypeError:
            return orig(json)  # exotic types orjson can't handle → stdlib
        headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        return headers, ByteStream(body)

    # Verify our output matches the real httpx contract (types + headers) on a
    # tiny probe before swapping; otherwise leave stdlib in place.
    try:
        ref_headers, ref_stream = orig({"_rllm_probe": 1})
        new_headers, new_stream = _encode_json({"_rllm_probe": 1})
        if type(new_stream) is not type(ref_stream) or set(new_headers) != set(ref_headers):
            raise ValueError("httpx encode_json shape mismatch")
    except Exception as e:  # noqa: BLE001
        logger.warning("httpx orjson patch skipped (%s); using stdlib json for httpx bodies", e)
        return

    _encode_json._rllm_orjson_patch = True  # type: ignore[attr-defined]
    _hc.encode_json = _encode_json
    logger.info("Installed httpx orjson encode_json patch (faster large-prompt serialization off the gateway loop's critical path)")


_install_httpx_orjson_patch()


def _install_sdk_response_parse_patch() -> None:
    """Parse the SDK's streaming completion chunks with orjson.

    ``DeploymentSampler.async_completions_stream`` parses every SSE chunk with
    stdlib ``json.loads(sse.data)`` on the event-loop thread; across many
    concurrent streams that per-chunk parse is a continuous on-loop cost. Swap
    the SDK ``sampling`` module's ``json`` reference for a shim whose ``loads`` is
    orjson and which delegates every other attribute (``dumps``,
    ``JSONDecodeError``, …) to stdlib json — so nothing else in the module
    changes behaviour. ``orjson.JSONDecodeError`` subclasses ``ValueError``, so
    the SDK's ``except (ValueError, TypeError)`` still catches malformed chunks.

    Best-effort and idempotent: skips silently if the SDK/orjson layout differs.
    """
    try:
        import json as _stdlib_json

        import orjson
        from fireworks.training.sdk import sampling as _sdk
    except Exception:  # noqa: BLE001 - layout unknown → skip
        return

    if getattr(_sdk, "_rllm_orjson_json_shim", False):
        return

    class _OrjsonJson:
        loads = staticmethod(orjson.loads)  # the hot path

        def __getattr__(self, name):  # noqa: ANN001 - delegate everything else
            return getattr(_stdlib_json, name)

    shim = _OrjsonJson()
    try:
        assert shim.loads('{"a":1}') == {"a": 1}
        assert shim.dumps({"a": 1}) == '{"a": 1}'  # delegates to stdlib → str
    except Exception as e:  # noqa: BLE001
        logger.warning("SDK response-parse orjson patch skipped (%s); using stdlib json", e)
        return

    _sdk.json = shim
    _sdk._rllm_orjson_json_shim = True  # type: ignore[attr-defined]
    logger.info("Installed orjson patch for Fireworks SDK streaming response parse")


_install_sdk_response_parse_patch()


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
        bypass_render_with_parser: bool = False,
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
        #
        # bypass_render_with_parser=True is an explicit escape hatch: force
        # ChatTemplateParser and skip the unified renderer entirely.
        if bypass_render_with_parser:
            self.unified_renderer = None
        else:
            from rllm.renderers import resolve

            res = resolve(
                getattr(tokenizer, "name_or_path", None),
                tokenizer,
                backend="fireworks",
                family=renderer_family,
                renderer_name=renderer_name,
            )
            explicit = renderer_name is not None or renderer_family != "auto"
            self.unified_renderer = res.renderer if (res.source == "tinker" or (explicit and res.source != "chat_template")) else None
            if self.unified_renderer is not None:
                logger.info("FireworksEngine rendering via %s renderer (%s)", res.name, res.source)
            elif explicit and res.source == "chat_template":
                logger.warning(
                    "renderer_family=%r / renderer_name=%r did not resolve a native renderer for %s; using ChatTemplateParser.",
                    renderer_family,
                    renderer_name,
                    getattr(tokenizer, "name_or_path", None),
                )

        # No tinker_cookbook renderer on this path; the unified renderer (above) or
        # chat_parser owns rendering+parsing — both handled by the shared TinkerEngine.
        # ``renderer`` stays None so the shared legacy/VLM branch is never reached here.
        self.renderer = None
        # bypass_render_with_parser reflects who owns rendering+parsing: True =
        # ChatTemplateParser (built below); False = the unified renderer.
        self.bypass_render_with_parser = self.unified_renderer is None
        self.chat_parser = None if self.unified_renderer is not None else ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=disable_thinking)

        self.sample_timeout = sample_timeout
        self.router_replay = router_replay
        self.sampling_client = sampler
        # Retained so a separate-process gateway can rebuild an equivalent engine
        # (see handler_factory_spec / rllm.gateway.worker_handlers).
        self.renderer_family = renderer_family

    def handler_factory_spec(self) -> tuple[str, dict[str, Any]]:
        """Recipe for rebuilding this engine as a gateway ``local_handler`` in a
        separate process (Path 1 / rllm.gateway.manager multi-process mode).

        Returns ``(import_path, config)`` where ``import_path`` is a
        ``"module:function"`` that maps ``config`` -> a ``local_handler``. The
        config carries only serializable, non-secret values — the subprocess
        attaches to the *same* Fireworks deployment via the sampler's
        ``base_url``/``model`` (no re-provisioning); the API key comes from the
        inherited ``FIREWORKS_API_KEY`` env, not this config.
        """
        sampler = self.sampling_client
        return (
            "rllm.gateway.worker_handlers:build_fireworks_handler",
            {
                "inference_url": getattr(sampler, "base_url", None),
                "model": getattr(sampler, "model", None),
                "tokenizer_model": getattr(self.tokenizer, "name_or_path", None),
                "max_prompt_length": self.max_prompt_length,
                "max_response_length": self.max_response_length,
                # __init__ stored (input - 1); pass +1 so a rebuild lands identically.
                "max_model_length": self.max_model_length + 1,
                "sampling_params": {"train": self.train_sampling_params, "val": self.val_sampling_params},
                "reasoning_effort": self.reasoning_effort,
                "accumulate_reasoning": self.accumulate_reasoning,
                "router_replay": self.router_replay,
                "sample_timeout": self.sample_timeout,
                "renderer_family": self.renderer_family,
                "bypass_render_with_parser": self.bypass_render_with_parser,
            },
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

        # Rendering a ≤120K-token prompt is CPU-bound and (via the HF fast
        # tokenizer) releases the GIL, so run it in a worker thread instead of on
        # the gateway's single event loop — otherwise every turn-0 request blocks
        # the loop from flushing responses / firing heartbeats for all other
        # in-flight requests, which at high concurrency shows up as client
        # timeouts + TokenAccumulator "duplicate" churn.
        loop = asyncio.get_running_loop()
        token_input = await loop.run_in_executor(
            None,
            functools.partial(
                self._render_prompt_token_input,
                messages,
                tools=tools,
                reasoning_effort=reasoning_effort,
                accumulate_reasoning=accumulate_reasoning,
            ),
        )

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
        # trajectory's turns. Fireworks accepts either affinity header, and the
        # OpenAI ``user`` field is also used for routing; keep all three equal.
        session_headers = None
        if session_id:
            session_id = str(session_id)
            sampling_params["user"] = session_id
            session_headers = {
                "x-multi-turn-session-id": session_id,
                "x-session-affinity": session_id,
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

        start = time.monotonic()
        first_failure: float | None = None
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
                # Timeouts/transport errors have an empty str(exc), so the string
                # markers below miss them; classify by type instead. Raw
                # ssl.SSLError (e.g. bad_record_mac) escapes the SDK's SSE
                # stream unwrapped by httpx.
                is_network_error = isinstance(exc, httpx.TimeoutException | httpx.TransportError | ssl.SSLError)
                transient = isinstance(exc, _EmptyCompletionIdsError) or is_network_error or any(marker in err or marker in exc_name for marker in _TRANSIENT_ERROR_MARKERS)
                elapsed = time.monotonic() - start
                if first_failure is None:
                    first_failure = time.monotonic()
                wait = min(10 * (attempt + 1), _RETRY_BACKOFF_CAP_S)
                # Retry only while there's budget left to both back off and make
                # another attempt worthwhile — else fail fast so the held client
                # connection is released instead of stalled past its tolerance.
                # The budget clock starts at the FIRST FAILURE, not request start:
                # a long healthy generation that dies mid-stream must not have its
                # own streaming time charged against the retry budget.
                budget_left = (time.monotonic() - first_failure) + wait < _RETRY_BUDGET_S
                if transient and attempt < _MAX_SAMPLE_ATTEMPTS - 1 and budget_left:
                    logger.debug(
                        "Attempt %d/%d failed (%s: %s) after %.1fs, retrying in %ds...",
                        attempt + 1,
                        _MAX_SAMPLE_ATTEMPTS,
                        exc_name,
                        exc,
                        elapsed,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp = getattr(exc, "response", None)
                resp_text = getattr(resp, "text", None)
                resp_headers = dict(getattr(resp, "headers", None) or {})
                give_up = "retry budget exhausted" if (transient and not budget_left) else "permanent"
                logger.error(
                    "Sampling failed (%s) after %d attempts / %.1fs (%s): %s\n%s\nheaders: %s",
                    give_up,
                    attempt + 1,
                    elapsed,
                    exc_name,
                    exc,
                    resp_text or "",
                    resp_headers,
                )
                raise
        raise RuntimeError("unreachable")
