"""httpx-based reverse proxy with streaming SSE support.

Reference: miles ``MilesRouter._do_proxy()``
(``miles/router/router.py`` lines 138-166).
"""

import asyncio
import functools
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from rllm_model_gateway import fastjson
from rllm_model_gateway.data_process import (
    build_trace_record,
    build_trace_record_from_chunks,
    extract_completion_token_ids,
    extract_logprobs,
    extract_prompt_token_ids,
    extract_routing_matrices,
    strip_vllm_fields,
)
from rllm_model_gateway.models import TraceRecord
from rllm_model_gateway.session_router import SessionRouter
from rllm_model_gateway.store.base import TraceStore
from rllm_model_gateway.token_accumulator import (
    ResetReason,
    TokenAccumulator,
)

logger = logging.getLogger(__name__)

# Headers that should not be forwarded verbatim
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "content-length",
        "content-encoding",
        "host",
    }
)

# A generation interrupted by a weight sync comes back as finish_reason="abort"
# and is resumed from its partial token IDs. Cap the resumes so a worker that
# keeps aborting (or returns no new tokens) fails the turn instead of looping.
_MAX_ABORT_RESUMES = 3
_RETRYABLE_HTTP_ERRORS = (
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)


def _strip_logprobs(response: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *response* with ``logprobs`` removed from each choice.

    Called when the gateway injected ``logprobs=True`` but the original
    client request did not ask for them — keeps the proxy transparent.

    Returns a new dict so that the original (used for trace capture) is
    never mutated.
    """
    if "choices" not in response:
        return response
    return {
        **response,
        "choices": [{k: v for k, v in choice.items() if k != "logprobs"} for choice in response["choices"]],
    }


def _finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def _merge_resumed_response(
    response: dict[str, Any],
    resumed: dict[str, Any],
    *,
    chat_response: bool,
    prompt_token_ids: list[int],
) -> dict[str, Any]:
    """Append one completions response to an aborted response."""
    choices = response.get("choices") or []
    resumed_choices = resumed.get("choices") or []
    if not choices or not resumed_choices:
        return response

    choice = choices[0]
    resumed_choice = resumed_choices[0]
    prior_ids = extract_completion_token_ids(response)
    resumed_ids = extract_completion_token_ids(resumed)
    prior_logprobs = extract_logprobs(response)
    resumed_logprobs = extract_logprobs(resumed)

    if chat_response:
        message = choice.setdefault("message", {"role": "assistant", "content": ""})
        message["content"] = (message.get("content") or "") + (resumed_choice.get("text") or "")
        if prior_logprobs or resumed_logprobs:
            choice["logprobs"] = {"content": [{"logprob": value} for value in prior_logprobs + resumed_logprobs]}
    else:
        choice["text"] = (choice.get("text") or "") + (resumed_choice.get("text") or "")
        if prior_logprobs or resumed_logprobs:
            choice["logprobs"] = {"token_logprobs": prior_logprobs + resumed_logprobs}

    prior_routing = extract_routing_matrices(response, len(prompt_token_ids))
    # A resumed chunk starts one token back, covering the token whose row the aborted chunk never got.
    resumed_routing = extract_routing_matrices(resumed, len(prompt_token_ids) + max(len(prior_ids) - 1, 0))
    choice["routed_experts"] = prior_routing + resumed_routing if prior_routing is not None and resumed_routing is not None else None

    choice["token_ids"] = prior_ids + resumed_ids
    choice["finish_reason"] = resumed_choice.get("finish_reason")
    if "stop_reason" in resumed_choice:
        choice["stop_reason"] = resumed_choice["stop_reason"]
    response["prompt_token_ids"] = prompt_token_ids

    usage = response.setdefault("usage", {})
    completion_tokens = len(choice["token_ids"])
    usage["prompt_tokens"] = len(prompt_token_ids)
    usage["completion_tokens"] = completion_tokens
    usage["total_tokens"] = len(prompt_token_ids) + completion_tokens
    return response


def _to_openai_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Parsed tool calls -> OpenAI chat ``tool_calls`` shape.

    Three producer shapes reach here, and reading the wrong level yields a
    structurally-present but empty tool call (name="") that OpenAI
    function-calling clients silently ignore:

    * ``renderers.ParsedToolCall`` dataclass, from ``parse_response``: flat
      ``.name`` / ``.arguments`` / ``.id`` attributes.
    * nested dict ``{"function": {"name", "arguments"}}``.
    * flat dict ``{"name", "arguments"}``.

    A call without a name is dropped: OpenAI's shape has no way to express
    "the model tried to call something and we couldn't tell what", and a
    ``name=""`` entry makes clients silently take no action. ``ParsedToolCall.status``
    is deliberately NOT used for that decision — Qwen3.5 emits XML call bodies
    (``<function=bash><parameter=command>``), which the renderer flags
    ``INVALID_JSON`` even though it recovered name and arguments cleanly.

    The OpenAI wire shape a client (e.g. opencode's OpenAI-compatible provider)
    expects is ``{"id", "type": "function", "index", "function": {"name",
    "arguments": <json str>}}``.
    """
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") if isinstance(tc, dict) else None
        if isinstance(fn, dict):
            name, args, call_id = fn.get("name", ""), fn.get("arguments", {}), tc.get("id")
        elif isinstance(tc, dict):
            name, args, call_id = tc.get("name", ""), tc.get("arguments", {}), tc.get("id")
        else:
            name, args, call_id = getattr(tc, "name", ""), getattr(tc, "arguments", {}), getattr(tc, "id", None)
            if not name:
                continue
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {})
        out.append(
            {
                "id": call_id or f"call_{i}",
                "type": "function",
                "index": i,
                "function": {"name": name or "", "arguments": args},
            }
        )
    return out


def _assistant_message_from_completion(
    choice: dict[str, Any],
    completion_token_ids: list[int] | None,
    request_body: dict[str, Any],
    renderer: Any,
) -> dict[str, Any]:
    """Build the assistant chat ``message`` for a cumulative-mode turn.

    Two producers feed the cumulative-turn path:

    * The in-process handler (Fireworks/Tinker ``local_handler``) already parsed the
      completion with the engine's renderer, so its choice carries structured
      ``tool_calls`` / ``reasoning`` (see ``rllm.gateway.tinker_adapter``). Pass them
      straight through — this is the source of truth.
    * A raw vLLM ``/v1/completions`` worker (verl backend) returns only ``text``. When
      the client sent ``tools`` (OpenAI function-calling clients such as opencode), parse
      the completion tokens via the renderer into structured tool_calls; the
      ``/v1/completions`` bridge skips the serving stack's own chat tool-call parser, so
      without this such clients get only raw ``<tool_call>…`` text and take no action.

    Text-protocol harnesses (Terminus-2) send no ``tools`` and get the raw text (unchanged).
    Only the CLIENT-facing message is shaped; training token capture uses the raw
    ``completion_token_ids`` and is untouched.
    """
    text = choice.get("text", "") if isinstance(choice, dict) else ""

    # (a) Handler already produced structured fields — pass through.
    if isinstance(choice, dict) and (choice.get("tool_calls") or choice.get("reasoning")):
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if choice.get("reasoning"):
            message["reasoning"] = choice["reasoning"]
        if choice.get("tool_calls"):
            message["tool_calls"] = choice["tool_calls"]
        return message

    # (b) Raw completion (HTTP worker): parse tokens when the client wants tool calls.
    message = {"role": "assistant", "content": text}
    if not request_body.get("tools"):
        return message
    parse = getattr(renderer, "parse_response", None)
    if not callable(parse) or not completion_token_ids:
        return message
    try:
        parsed = parse(list(completion_token_ids))
    except Exception:
        logger.warning("cumulative-turn tool-call parse failed; returning raw content", exc_info=True)
        return message
    tool_calls = _to_openai_tool_calls(parsed.tool_calls) if getattr(parsed, "tool_calls", None) else None
    reasoning = getattr(parsed, "reasoning_content", None)
    if not tool_calls and not reasoning:
        return message  # nothing structured recovered -> keep raw text (defensive)
    message["content"] = parsed.content or ""
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _build_trace_data(
    session_id: str,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    latency_ms: float,
    weight_version: int | None,
    capture_raw: bool,
    metadata: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any]]:
    """Build a TraceRecord and serialize it to a dict.

    Runs in a worker thread (see ``ReverseProxy._persist_trace``) so the token-id
    list copies + ``model_dump`` stay off the event-loop thread. Reads
    ``request_body``/``response_body`` without mutating them, so it's safe to run
    concurrently with the response path.
    """
    trace = build_trace_record(
        session_id,
        request_body,
        response_body,
        latency_ms,
        metadata=metadata,
        weight_version=weight_version,
        capture_raw=capture_raw,
    )
    return trace.trace_id, trace.session_id, trace.model_dump()


def _context_limit_metadata(
    request_body: dict[str, Any],
    completions_body: dict[str, Any],
) -> dict[str, Any]:
    requested = request_body.get("max_tokens")
    effective = completions_body.get("max_tokens")
    if not isinstance(requested, int) or not isinstance(effective, int) or effective >= requested:
        return {}
    return {
        "max_tokens_clamped": True,
        "requested_max_tokens": requested,
        "effective_max_tokens": effective,
    }


class ReverseProxy:
    """Forward requests to inference workers, capture traces.

    Non-streaming requests are fully buffered so that the complete response
    can be inspected for token IDs and logprobs.

    Streaming (SSE) requests are forwarded chunk-by-chunk in real time.
    Chunks are buffered internally so that a ``TraceRecord`` can be assembled
    after ``[DONE]``.
    """

    def __init__(
        self,
        router: SessionRouter,
        store: TraceStore,
        *,
        strip_vllm: bool = True,
        sync_traces: bool = False,
        max_retries: int = 2,
        local_handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        cumulative_token_mode: bool = False,
        renderer: Any = None,
        max_model_len: int | None = None,
        heartbeat_initial_delay_s: float = 50.0,
        heartbeat_interval_s: float = 25.0,
        heartbeat_budget_s: float = 3600.0,
        capture_raw_payloads: bool = False,
        loop_health_enabled: bool = False,
        worker_label: str = "",
    ) -> None:
        self.router = router
        self.store = store
        # Distinguishes this process's logs when several gateway workers run
        # behind a front (num_workers>1); empty for the single-process case.
        self.worker_label = f"[{worker_label}] " if worker_label else ""
        self.strip_vllm = strip_vllm
        self.sync_traces = sync_traces
        self.max_retries = max_retries
        self.local_handler = local_handler
        self.cumulative_token_mode = cumulative_token_mode
        self.renderer = renderer
        self.max_model_len = max_model_len
        # Retain full raw request/response on each trace. Off by default: training
        # reads only token-id/logprob/message fields, and serializing the raw
        # dicts (≤120K-token prompt + full response) is the dominant per-request
        # CPU cost on the event loop at high concurrency. Enable for debugging.
        self.capture_raw_payloads = capture_raw_payloads
        self.loop_health_enabled = loop_health_enabled
        # Whitespace heartbeat for slow non-streaming completions: middleboxes
        # on the response path (Cloudflare quick tunnel: 120s; ngrok: ~300s;
        # NAT flow tables) silently kill responses that stay byte-silent while
        # the model generates. After ``initial_delay`` with no upstream result,
        # the response is committed as chunked and a single space (legal JSON
        # leading whitespace, invisible to every JSON parser) is emitted every
        # ``interval`` until the real body is ready. ``interval <= 0`` disables.
        self.heartbeat_initial_delay_s = heartbeat_initial_delay_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self.heartbeat_budget_s = heartbeat_budget_s
        self.weight_version: int | None = None
        self._http: httpx.AsyncClient | None = None
        self._pending_traces: set[asyncio.Task[None]] = set()
        self._accumulators: dict[str, TokenAccumulator] = {}
        # Loop-health instrumentation (diagnostic only; see _loop_health_monitor).
        # _inflight counts concurrent in-flight generations; _recent_lag_ms is the
        # latest event-loop lag sample, stamped onto duplicate logs for correlation.
        self._inflight: int = 0
        self._inflight_max: int = 0
        self._recent_lag_ms: float = 0.0
        self._monitor_task: asyncio.Task[None] | None = None

    def _get_accumulator(self, session_id: str) -> TokenAccumulator:
        """Return the TokenAccumulator for *session_id*, creating if needed."""
        if session_id not in self._accumulators:
            self._accumulators[session_id] = TokenAccumulator(self.renderer, session_id=session_id)
        return self._accumulators[session_id]

    def _track_inflight(self, task: asyncio.Future) -> None:
        """Count *task* (an in-flight generation) toward the in-flight gauge until
        it completes. Called by ``_respond_with_heartbeat`` around the result task
        so the gauge reflects concurrent generation load regardless of whether the
        response is buffered or heartbeat-streamed."""
        self._inflight += 1
        if self._inflight > self._inflight_max:
            self._inflight_max = self._inflight

        def _done(_: asyncio.Future) -> None:
            self._inflight -= 1

        task.add_done_callback(_done)

    async def _loop_health_monitor(self, sample_s: float = 0.5, report_s: float = 20.0) -> None:
        """Log event-loop health so the single-loop gateway's headroom is
        observable under load. Diagnostic only — no behavioural effect.

        ``lag`` is how much longer than ``sample_s`` a bare sleep actually took —
        i.e. how long the loop couldn't run this callback, whether from on-loop
        CPU or from GIL starvation by the trainer thread. ``thread_cpu`` is the
        loop thread's own CPU utilisation over the window, which disambiguates
        the two: high lag + high thread_cpu = self-CPU bound (offload / do less);
        high lag + low thread_cpu = the loop thread is starved (a separate gateway
        process would help). ``inflight`` is concurrent generations.
        """
        lags: list[float] = []
        window_start = time.monotonic()
        last_cpu = time.thread_time()
        next_report = window_start + report_s
        while True:
            t0 = time.monotonic()
            try:
                await asyncio.sleep(sample_s)
            except asyncio.CancelledError:
                return
            lag_ms = max(0.0, (time.monotonic() - t0 - sample_s) * 1000.0)
            self._recent_lag_ms = lag_ms
            lags.append(lag_ms)
            now = time.monotonic()
            if now >= next_report and lags:
                window = now - window_start
                cpu = time.thread_time()
                util = 100.0 * (cpu - last_cpu) / window if window > 0 else 0.0
                ordered = sorted(lags)
                p50 = ordered[len(ordered) // 2]
                p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
                logger.info(
                    "%sgateway loop health: lag_ms p50=%.0f p99=%.0f max=%.0f | thread_cpu=%.0f%% | inflight cur=%d max=%d | window=%.0fs",
                    self.worker_label,
                    p50,
                    p99,
                    ordered[-1],
                    util,
                    self._inflight,
                    self._inflight_max,
                    window,
                )
                lags.clear()
                self._inflight_max = self._inflight
                last_cpu = cpu
                window_start = now
                next_report = now + report_s

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=None),  # no timeout — LLM calls can be long
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
            follow_redirects=True,
        )
        if self.loop_health_enabled and self._monitor_task is None:
            self._monitor_task = asyncio.ensure_future(self._loop_health_monitor())

    async def stop(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None
        # Drain pending trace writes before closing
        if self._pending_traces:
            logger.info("Draining %d pending trace writes...", len(self._pending_traces))
            await asyncio.gather(*self._pending_traces, return_exceptions=True)
            self._pending_traces.clear()
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._http is None:
            await self.start()

    async def handle(self, request: Request) -> Response:
        """Proxy *request* to an inference worker, capture trace, return response."""
        request.state.weight_version = self.weight_version
        await self._ensure_started()
        session_id: str | None = request.state.session_id
        originally_requested_logprobs: bool = getattr(request.state, "originally_requested_logprobs", False)
        body = await request.body()

        try:
            request_body = fastjson.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            request_body = {}

        is_stream = request_body.get("stream", False)

        # Cumulative token mode interception: if enabled and past first turn,
        # rewrite to /v1/completions with pre-tokenized prompt to avoid drift.
        if self.cumulative_token_mode and session_id and request.url.path.endswith("/chat/completions"):
            acc = self._get_accumulator(session_id)
            if acc.should_rewrite():
                messages = request_body.get("messages", [])
                # Classify the incoming request against the accumulated snapshot.
                # plan.action is "extend" (build a /v1/completions bridge) or
                # "reset" (fall through and re-render as a fresh turn-0). Every
                # reset is logged with its classified ResetReason + diagnostics so
                # the *why* is recoverable from the gateway log.
                plan = acc.plan_turn(messages)
                if plan.action == "extend":
                    # bridge_to_next_turn renders the new messages and concatenates
                    # ≤120K prior token ids — CPU-bound, so keep it off the single
                    # event loop (the tokenizer half releases the GIL) so the loop
                    # stays free to service other concurrent requests.
                    loop = asyncio.get_running_loop()
                    token_ids = await loop.run_in_executor(
                        None,
                        functools.partial(acc.build_next_prompt, plan.new_messages, tools=request_body.get("tools")),
                    )
                    if token_ids is not None:
                        logger.debug(
                            "TokenAccumulator extend session=%s turn=%d +%d new msgs",
                            session_id,
                            acc.turn_count,
                            len(plan.new_messages),
                        )
                        return await self._handle_cumulative_turn(
                            request,
                            request_body,
                            session_id,
                            acc,
                            token_ids,
                            originally_requested_logprobs,
                        )
                    # Structurally a valid extension, but the renderer couldn't
                    # prove the prefix-extension contract (e.g. DefaultRenderer).
                    # Reset so this turn is re-ingested as a fresh turn-0;
                    # otherwise the stale prefix would drop this turn's completion
                    # tokens from the next cumulative prompt and break extension.
                    acc.reset(
                        ResetReason.RENDERER_NO_BRIDGE,
                        diagnostics={
                            **plan.diagnostics,
                            "renderer": type(acc.renderer).__name__,
                            "new_roles": [m.get("role") for m in plan.new_messages],
                        },
                    )
                elif plan.action == "replay" and acc.prev_prompt_ids:
                    # Duplicate resend (upstream retry): the conversation did not
                    # advance. Regenerate from the same prompt and overwrite this
                    # turn in place instead of resetting — keeps drift-free state
                    # and avoids a spurious segment break for a single step. A
                    # fresh sample (not a cached one) is returned, so a retry that
                    # depends on resampling still makes progress.
                    logger.info(
                        "%sTokenAccumulator duplicate session=%s turn=%d age_s=%s inflight=%d loop_lag_ms=%.0f: regenerating in place, no reset",
                        self.worker_label,
                        session_id,
                        acc.turn_count,
                        plan.diagnostics.get("age_s", "?"),
                        self._inflight,
                        self._recent_lag_ms,
                    )
                    return await self._handle_cumulative_turn(
                        request,
                        request_body,
                        session_id,
                        acc,
                        list(acc.prev_prompt_ids),
                        originally_requested_logprobs,
                        replay=True,
                    )
                else:
                    acc.reset(plan.reason, diagnostics=plan.diagnostics)

            # Opening turn (or the turn right after a reset): nothing to bridge
            # from, so render the messages from scratch and take the same
            # /v1/completions path. That keeps BOTH the rendering and the
            # tool-call / reasoning extraction on the renderer for every turn of
            # a session — a raw chat response from an HTTP worker carries neither
            # unless the serving stack happens to run its own parsers. An
            # in-process handler already parses what it returns, so it stays on
            # the chat path.
            if self.local_handler is None and acc.turn_count == 0:
                loop = asyncio.get_running_loop()
                try:
                    token_ids = await loop.run_in_executor(
                        None,
                        functools.partial(
                            acc.build_initial_prompt,
                            request_body.get("messages", []),
                            tools=request_body.get("tools"),
                        ),
                    )
                except Exception:
                    logger.warning("%sfirst-turn render failed; using the chat path", self.worker_label, exc_info=True)
                    token_ids = None
                if token_ids:
                    return await self._handle_cumulative_turn(
                        request,
                        request_body,
                        session_id,
                        acc,
                        token_ids,
                        originally_requested_logprobs,
                    )

        if is_stream:
            return await self._handle_streaming(request, body, request_body, session_id, originally_requested_logprobs)
        return await self._handle_non_streaming(request, body, request_body, session_id, originally_requested_logprobs)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def _handle_non_streaming(
        self,
        request: Request,
        raw_body: bytes,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool = False,
    ) -> Response:
        """Proxy a non-streaming request, keeping the response connection warm.

        The upstream call runs as a task; if it finishes within
        ``heartbeat_initial_delay_s`` the plain JSON response goes out with its
        true status code (fast successes AND fast upstream errors are
        untouched). Past that, the response is committed as chunked 200 and a
        space is emitted every ``heartbeat_interval_s`` so no middlebox on the
        path (tunnel edge read timers, NAT flow tables) sees a byte-silent
        connection while the model generates. Leading spaces are insignificant
        JSON whitespace — parsed output is byte-identical for every client.
        """
        return await self._respond_with_heartbeat(self._non_streaming_result(request, raw_body, request_body, session_id, originally_requested_logprobs))

    async def _respond_with_heartbeat(self, result_coro: Awaitable[tuple[bytes, int]]) -> Response:
        """Await *result_coro* (returns ``(content_bytes, status_code)``); keep
        the client connection warm if it's slow.

        If the coroutine finishes within ``heartbeat_initial_delay_s`` the plain
        JSON response goes out with its true status code. Past that, the response
        is committed as chunked 200 and a space is emitted every
        ``heartbeat_interval_s`` so no middlebox on the path (tunnel edge read
        timers, NAT flow tables) sees a byte-silent connection while the model
        generates. Leading spaces are insignificant JSON whitespace — parsed
        output is byte-identical for every client. Shared by the turn-0 chat path
        and the cumulative (turn 1+) path so long generations survive on both.
        """
        result_task = asyncio.ensure_future(result_coro)
        self._track_inflight(result_task)
        if self.heartbeat_interval_s <= 0:
            content, status_code = await result_task
            return Response(content=content, status_code=status_code, media_type="application/json")

        try:
            content, status_code = await asyncio.wait_for(asyncio.shield(result_task), timeout=self.heartbeat_initial_delay_s)
            return Response(content=content, status_code=status_code, media_type="application/json")
        except TimeoutError:
            pass  # slow generation — switch to the heartbeat stream
        except asyncio.TimeoutError:  # noqa: UP041 — pre-3.11 alias, kept for safety
            pass

        async def _heartbeat_stream():
            deadline = time.monotonic() + self.heartbeat_budget_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result_task.cancel()
                    logger.error("Heartbeat budget (%.0fs) exhausted waiting for upstream; failing the request", self.heartbeat_budget_s)
                    yield json.dumps(
                        {"error": {"message": f"gateway heartbeat budget of {self.heartbeat_budget_s:.0f}s exhausted waiting for upstream", "type": "gateway_upstream_timeout", "code": 504}}
                    ).encode()
                    return
                try:
                    content, status_code = await asyncio.wait_for(asyncio.shield(result_task), timeout=min(self.heartbeat_interval_s, remaining))
                except (TimeoutError, asyncio.TimeoutError):
                    yield b" "  # legal JSON leading whitespace; resets every idle timer on the path
                    continue
                except Exception as e:  # noqa: BLE001 — surface upstream failure as a parseable error body
                    logger.warning("Upstream failed after heartbeat commit (status already 200): %s", e)
                    yield json.dumps({"error": {"message": f"gateway upstream failure: {e}", "type": "gateway_upstream_error", "code": 502}}).encode()
                    return
                if status_code != 200:
                    # Status line is already committed as 200; the upstream error
                    # body still goes through — clients see a JSON error object
                    # instead of a typed status, and the log keeps the truth.
                    logger.warning("Upstream returned %d after heartbeat commit; forwarding error body under 200", status_code)
                yield content
                return

        return StreamingResponse(_heartbeat_stream(), status_code=200, media_type="application/json")

    async def _non_streaming_result(
        self,
        request: Request,
        raw_body: bytes,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool = False,
    ) -> tuple[bytes, int]:
        t0 = time.perf_counter()

        if self.local_handler is not None:
            # In-process path: call handler directly, no HTTP.
            # Forward the rollout session id so affinity-aware engines (Fireworks)
            # can pin a trajectory's turns to one replica for prefix-cache reuse.
            if session_id:
                request_body["rllm_session_id"] = session_id
            response_body = await self.local_handler(request_body)
            status_code = 200
        else:
            # HTTP proxy path
            worker = self.router.route(session_id)
            url = self._build_url(worker.api_url, request.url.path, str(request.url.query))
            headers = self._forward_headers(request, session_id)
            try:
                response_body, status_code = await self._send_with_abort_resume(
                    method=request.method,
                    url=url,
                    content=raw_body,
                    headers=headers,
                    request_body=request_body,
                    worker_url=worker.api_url,
                    chat_response=request.url.path.endswith("/chat/completions"),
                )
            finally:
                self.router.release(worker.url)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist trace
        if session_id and response_body:
            await self._persist_trace(session_id, request_body, response_body, latency_ms, request.state.weight_version)

            # Ingest first turn into accumulator for cumulative token mode
            if self.cumulative_token_mode and request.url.path.endswith("/chat/completions"):
                acc = self._get_accumulator(session_id)
                if acc.turn_count == 0:
                    prompt_ids = extract_prompt_token_ids(response_body)
                    completion_ids = extract_completion_token_ids(response_body)
                    if prompt_ids or completion_ids:
                        acc.ingest_turn(prompt_ids, completion_ids)
                        acc.update_prefix(request_body.get("messages", []))

        # Sanitise response
        needs_strip_vllm = self.strip_vllm
        needs_strip_logprobs = not originally_requested_logprobs

        sanitized = response_body
        if isinstance(response_body, dict) and response_body:
            if needs_strip_vllm:
                sanitized = strip_vllm_fields(response_body)
            if needs_strip_logprobs:
                sanitized = _strip_logprobs(sanitized)

        return fastjson.dumps(sanitized), status_code

    # ------------------------------------------------------------------
    # Cumulative token mode
    # ------------------------------------------------------------------

    def _clamp_max_tokens(self, body: dict[str, Any], prompt_len: int) -> None:
        """Shrink ``max_tokens`` to the prompt's remaining context headroom.

        vLLM's OpenAI server rejects ``prompt_tokens + max_tokens >
        max_model_len`` with a 400 rather than truncating the request, so a
        cumulative prompt within the window still fails once the per-turn cap
        pushes it over. VERL's own token-in-token-out path clamps identically
        (``vllm_async_server.generate``: ``max(1, min(max_tokens,
        max_model_len - len(prompt_ids)))``); the floor of 1 exists because
        vLLM ≥0.20 raises on ``max_tokens < 1``. A prompt at or past the window
        still 400s, which is the same outcome VERL produces.
        """
        requested = body.get("max_tokens")
        if not self.max_model_len or not isinstance(requested, int):
            return
        clamped = max(1, min(requested, self.max_model_len - prompt_len))
        if clamped != requested:
            body["max_tokens"] = clamped
            logger.debug(
                "%sclamped max_tokens %d -> %d (prompt=%d, max_model_len=%d)",
                self.worker_label,
                requested,
                clamped,
                prompt_len,
                self.max_model_len,
            )

    async def _handle_cumulative_turn(
        self,
        request: Request,
        request_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
        originally_requested_logprobs: bool = False,
        *,
        replay: bool = False,
    ) -> Response:
        """Rewrite chat/completions to /v1/completions with pre-tokenized prompt.

        ``token_ids`` is the full prompt for this turn, built in ``handle()`` by
        ``acc.build_next_prompt`` (bridge-extended) or ``acc.build_initial_prompt``
        (opening turn) — or, when ``replay`` is set, the prior turn's prompt being
        regenerated after a duplicate resend.
        ``replay`` overwrites the current turn in place (``advance=False``)
        instead of recording a new one.

        Respects the original stream setting: if the client requested streaming,
        we stream from vLLM and translate completions chunks to chat format in
        real-time.
        """
        is_stream = request_body.get("stream", False)

        # Construct completions request: forward everything except chat-specific fields
        completions_body = {k: v for k, v in request_body.items() if k not in ("messages", "stream", "stream_options", "tools", "tool_choice")}
        completions_body["prompt"] = token_ids
        completions_body["add_special_tokens"] = False
        self._clamp_max_tokens(completions_body, len(token_ids))

        if is_stream:
            return await self._handle_cumulative_streaming(request, request_body, completions_body, session_id, acc, token_ids, replay=replay)
        return await self._handle_cumulative_non_streaming(
            request,
            request_body,
            completions_body,
            session_id,
            acc,
            token_ids,
            originally_requested_logprobs,
            replay=replay,
        )

    async def _handle_cumulative_non_streaming(
        self,
        request: Request,
        request_body: dict[str, Any],
        completions_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
        originally_requested_logprobs: bool = False,
        *,
        replay: bool = False,
    ) -> Response:
        """Non-streaming cumulative turn: send pre-tokenized prompt, return JSON.

        Wrapped in the shared whitespace heartbeat so a slow turn-1+ generation
        keeps its (otherwise byte-silent) client connection alive instead of
        being idle-cut and re-sent as a duplicate.
        """
        return await self._respond_with_heartbeat(
            self._cumulative_non_streaming_result(request, request_body, completions_body, session_id, acc, token_ids, originally_requested_logprobs, replay=replay)
        )

    async def _cumulative_non_streaming_result(
        self,
        request: Request,
        request_body: dict[str, Any],
        completions_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
        originally_requested_logprobs: bool = False,
        *,
        replay: bool = False,
    ) -> tuple[bytes, int]:
        """Produce the cumulative-turn response bytes + status code.

        Routes to the in-process ``local_handler`` (Tinker/Fireworks) when
        present, otherwise POSTs ``/v1/completions`` to a vLLM worker. Both
        return a completions-style body carrying ``prompt_token_ids`` +
        ``token_ids``. Runs as the heartbeat-wrapped result task, so its
        accumulator side effects (``ingest_turn`` / ``update_prefix``) complete
        exactly once even if the client connection drops mid-flight.
        """
        t0 = time.perf_counter()

        if self.local_handler is not None:
            # In-process path (Tinker): sample directly from the pre-tokenized
            # prompt; no HTTP worker, no re-tokenization.
            if session_id:
                completions_body["rllm_session_id"] = session_id
            response_body = await self.local_handler(completions_body)
            status_code = 200
        else:
            worker = self.router.route(session_id)
            url = self._build_url(worker.api_url, "/v1/completions", "")
            headers = self._forward_headers(request, session_id)
            raw_body = json.dumps(completions_body).encode()
            try:
                response_body, status_code = await self._send_with_abort_resume(
                    method="POST",
                    url=url,
                    content=raw_body,
                    headers=headers,
                    request_body=completions_body,
                    worker_url=worker.api_url,
                    chat_response=False,
                )
            finally:
                self.router.release(worker.url)

        latency_ms = (time.perf_counter() - t0) * 1000

        prompt_token_ids = extract_prompt_token_ids(response_body) or token_ids
        completion_token_ids = extract_completion_token_ids(response_body)

        acc.ingest_turn(prompt_token_ids, completion_token_ids, advance=not replay)
        acc.update_prefix(request_body.get("messages", []))

        # Translate to chat format. The handler (Fireworks/Tinker) may have parsed
        # tool_calls/reasoning onto the choice; _assistant_message_from_completion prefers
        # those, else parses tokens (raw vLLM worker) when the client sent tools.
        choices = response_body.get("choices") or []
        if choices:
            first_choice = choices[0]
            message = _assistant_message_from_completion(first_choice, completion_token_ids, request_body, acc.renderer)
            for k in ("text", "tool_calls", "reasoning"):
                first_choice.pop(k, None)  # completions-level fields now live in message
            first_choice["message"] = message
        response_body["object"] = "chat.completion"

        if session_id and response_body:
            await self._persist_trace(
                session_id,
                request_body,
                response_body,
                latency_ms,
                request.state.weight_version,
                metadata=_context_limit_metadata(request_body, completions_body),
            )

        sanitized = response_body
        if isinstance(response_body, dict) and response_body:
            if self.strip_vllm:
                sanitized = strip_vllm_fields(response_body)
            if not originally_requested_logprobs:
                sanitized = _strip_logprobs(sanitized)

        return fastjson.dumps(sanitized), status_code

    async def _handle_cumulative_streaming(
        self,
        request: Request,
        request_body: dict[str, Any],
        completions_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
        *,
        replay: bool = False,
    ) -> StreamingResponse:
        """Streaming cumulative turn: stream from vLLM, translate chunks to chat format.

        ``replay`` overwrites the current turn in place (duplicate resend) rather
        than recording a new one.
        """
        if self.local_handler is not None:
            # In-process backends (Tinker) don't stream; synthesize an SSE
            # stream from a single pre-tokenized completion call.
            return await self._handle_cumulative_streaming_local(request, request_body, completions_body, session_id, acc, token_ids, replay=replay)

        completions_body["stream"] = True

        worker = self.router.route(session_id)
        url = self._build_url(worker.api_url, "/v1/completions", "")
        headers = self._forward_headers(request, session_id)
        raw_body = json.dumps(completions_body).encode()

        assert self._http is not None
        upstream = self._http.stream(
            method="POST",
            url=url,
            content=raw_body,
            headers=headers,
        )
        retry_client: httpx.AsyncClient | None = None
        try:
            resp = await upstream.__aenter__()
        except _RETRYABLE_HTTP_ERRORS as first_exc:
            logger.warning(
                "Cumulative streaming connection error to %s (type=%s). Retrying.",
                url,
                type(first_exc).__name__,
            )
            retry_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=None),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                follow_redirects=True,
            )
            retry_upstream = retry_client.stream(
                method="POST",
                url=url,
                content=raw_body,
                headers=headers,
            )
            try:
                resp = await retry_upstream.__aenter__()
                upstream = retry_upstream
            except Exception:
                await retry_client.aclose()
                self.router.release(worker.url)
                raise

        t0 = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        # OpenAI function-calling clients (e.g. opencode) need structured tool_calls, which
        # require the full completion to parse. Buffer the stream and fake-stream a single
        # reconstructed message at the end; text-protocol clients (no tools) stream through
        # unchanged. See _assistant_message_from_completion.
        tools_mode = bool(request_body.get("tools"))

        def _build_trace():
            latency_ms = (time.perf_counter() - t0) * 1000
            return build_trace_record_from_chunks(
                session_id,
                request_body,
                chunks,
                latency_ms,
                metadata=_context_limit_metadata(request_body, completions_body),
                weight_version=request.state.weight_version,
                capture_raw=self.capture_raw_payloads,
            )

        async def event_generator():
            built_trace = None
            try:
                first_chunk_sent = False
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        if line and not tools_mode:
                            yield line + "\n"
                        continue

                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        if not tools_mode:
                            yield "data: [DONE]\n\n"
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    chunks.append(chunk)

                    if tools_mode:
                        continue  # buffer; structured message emitted after the stream

                    # Translate completions chunk → chat chunk
                    choices = chunk.get("choices", [])
                    chat_chunk: dict[str, Any] = {
                        "id": chunk.get("id", ""),
                        "object": "chat.completion.chunk",
                        "created": chunk.get("created", 0),
                        "model": chunk.get("model", ""),
                        "choices": [],
                    }
                    if choices:
                        c = choices[0]
                        delta: dict[str, Any] = {}
                        if not first_chunk_sent:
                            delta["role"] = "assistant"
                            first_chunk_sent = True
                        text = c.get("text", "")
                        if text:
                            delta["content"] = text
                        chat_chunk["choices"] = [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": c.get("finish_reason"),
                            }
                        ]
                    elif not chunk.get("usage"):
                        # Empty chunk with no usage either — nothing to forward
                        continue

                    if chunk.get("usage"):
                        chat_chunk["usage"] = chunk["usage"]

                    sanitized = strip_vllm_fields(chat_chunk) if self.strip_vllm else chat_chunk
                    yield f"data: {json.dumps(sanitized)}\n\n"

                # Tools client: reconstruct one structured chat message from the full
                # completion and fake-stream it (role → content+tool_calls → finish → DONE).
                if tools_mode and chunks:
                    built_trace = _build_trace()
                    # Raw vLLM stream: no per-choice tool_calls, so pass a text-only choice
                    # and let the renderer parse the buffered completion tokens.
                    joined_text = "".join((c.get("choices") or [{}])[0].get("text", "") for c in chunks)
                    message = _assistant_message_from_completion(
                        {"text": joined_text},
                        built_trace.completion_token_ids,
                        request_body,
                        acc.renderer,
                    )
                    cid = chunks[0].get("id", "")
                    created = chunks[0].get("created", 0)
                    model = chunks[0].get("model", "")
                    usage = next((c["usage"] for c in reversed(chunks) if c.get("usage")), {})
                    finish_reason = next((fr for c in reversed(chunks) for ch in (c.get("choices") or []) if (fr := ch.get("finish_reason"))), "stop")
                    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
                    delta = {"content": message["content"]}
                    for k in ("reasoning_content", "tool_calls"):
                        if k in message:
                            delta[k] = message[k]
                    role_chunk = {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
                    msg_chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                    fin_chunk = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": usage}
                    if self.strip_vllm:
                        msg_chunk = strip_vllm_fields(msg_chunk)
                    yield f"data: {json.dumps(role_chunk)}\n\n"
                    yield f"data: {json.dumps(msg_chunk)}\n\n"
                    yield f"data: {json.dumps(fin_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            finally:
                await upstream.__aexit__(None, None, None)
                if retry_client is not None:
                    await retry_client.aclose()
                self.router.release(worker.url)

                # Ingest accumulated token data (reuse the trace already built for the
                # tools-mode emission, so chunks are parsed once).
                if chunks:
                    trace = built_trace or _build_trace()
                    prompt_ids = trace.prompt_token_ids or token_ids
                    completion_ids = trace.completion_token_ids

                    acc.ingest_turn(prompt_ids, completion_ids, advance=not replay)
                    acc.update_prefix(request_body.get("messages", []))

                    task = asyncio.create_task(self._safe_store(trace.trace_id, trace.session_id, trace.model_dump()))
                    self._pending_traces.add(task)
                    task.add_done_callback(self._pending_traces.discard)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            status_code=resp.status_code,
        )

    async def _handle_cumulative_streaming_local(
        self,
        request: Request,
        request_body: dict[str, Any],
        completions_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
        *,
        replay: bool = False,
    ) -> StreamingResponse:
        """Cumulative streaming via the in-process handler (fake-streaming).

        The local handler returns a full completion in one shot; we ingest its
        token IDs into the accumulator, translate to chat format, and emit it as
        a synthesized SSE stream (role → content+tokens → finish → [DONE]).

        ``replay`` overwrites the current turn in place (duplicate resend) rather
        than recording a new one.
        """
        assert self.local_handler is not None
        t0 = time.perf_counter()
        if session_id:
            completions_body["rllm_session_id"] = session_id
        response_body = await self.local_handler(completions_body)
        latency_ms = (time.perf_counter() - t0) * 1000

        prompt_token_ids = extract_prompt_token_ids(response_body) or token_ids
        completion_token_ids = extract_completion_token_ids(response_body)
        acc.ingest_turn(prompt_token_ids, completion_token_ids, advance=not replay)
        acc.update_prefix(request_body.get("messages", []))

        choices = response_body.get("choices") or []
        choice0 = choices[0] if choices else {}
        finish_reason = choice0.get("finish_reason") or "stop"
        completion_logprobs = choice0.get("logprobs")

        # Prefer the handler's structured tool_calls/reasoning (parsed onto the choice);
        # else raw content. extra_delta carries the structured fields into the streamed
        # content chunk. See _assistant_message_from_completion.
        message = _assistant_message_from_completion(choice0, completion_token_ids, request_body, acc.renderer)
        for k in ("text", "tool_calls", "reasoning"):
            choice0.pop(k, None)  # completions-level fields now live in message
        content = message["content"]
        extra_delta = {k: message[k] for k in ("reasoning", "reasoning_content", "tool_calls") if k in message}

        if session_id and response_body:
            chat_body = dict(response_body)
            chat_body["object"] = "chat.completion"
            if chat_body.get("choices"):
                chat_body["choices"][0]["message"] = message
            trace = build_trace_record(
                session_id,
                request_body,
                chat_body,
                latency_ms,
                metadata=_context_limit_metadata(request_body, completions_body),
                weight_version=request.state.weight_version,
                capture_raw=self.capture_raw_payloads,
            )
            await self._persist(trace)

        chat_id = response_body.get("id", "chatcmpl-local")
        created = response_body.get("created", int(time.time()))
        model = response_body.get("model", "")
        usage = response_body.get("usage", {})

        def _sanitize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
            return strip_vllm_fields(chunk) if self.strip_vllm else chunk

        async def event_generator():
            def _sse(data: str) -> str:
                return f"data: {data}\n\n"

            yield _sse(
                json.dumps(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                    }
                )
            )
            yield _sse(
                json.dumps(
                    _sanitize_chunk(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": content, **extra_delta}, "finish_reason": None, "token_ids": completion_token_ids, "logprobs": completion_logprobs}],
                            "prompt_token_ids": prompt_token_ids,
                        }
                    )
                )
            )
            yield _sse(
                json.dumps(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        "usage": usage,
                    }
                )
            )
            yield _sse("[DONE]")

        return StreamingResponse(event_generator(), media_type="text/event-stream", status_code=200)

    # ------------------------------------------------------------------
    # Streaming (SSE)
    # ------------------------------------------------------------------

    async def _handle_streaming(
        self,
        request: Request,
        raw_body: bytes,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool = False,
    ) -> StreamingResponse:
        if self.local_handler is not None:
            return await self._handle_streaming_local(request_body, session_id, originally_requested_logprobs, request.state.weight_version)

        worker = self.router.route(session_id)
        url = self._build_url(worker.api_url, request.url.path, str(request.url.query))
        headers = self._forward_headers(request, session_id)

        assert self._http is not None
        upstream = self._http.stream(
            method=request.method,
            url=url,
            content=raw_body,
            headers=headers,
        )
        # Retry is needed because pooled TCP connections can go stale during the
        # weight-update idle window: VPC silently drops idle sockets, and the next
        # request on that socket fails with httpx.ReadError / RemoteProtocolError
        # ("Server disconnected without sending a response") / ConnectError.
        # Without retry, these transient failures propagate as failed rollouts and
        # surface as ASGI exceptions in the agent loop.  The retry uses a fresh
        # single-use client (no pool) so it cannot hit another stale socket.
        # retry_client is non-None only when we fell back; event_generator's
        # finally block closes it after streaming completes.
        retry_client: httpx.AsyncClient | None = None
        try:
            resp = await upstream.__aenter__()
        except _RETRYABLE_HTTP_ERRORS as first_exc:
            logger.warning(
                "Connection error to %s (type=%s, msg=%s). Retrying with a fresh connection.",
                url,
                type(first_exc).__name__,
                first_exc,
            )

            retry_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=None),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                follow_redirects=True,
            )
            retry_upstream = retry_client.stream(
                method=request.method,
                url=url,
                content=raw_body,
                headers=headers,
            )
            try:
                resp = await retry_upstream.__aenter__()
                upstream = retry_upstream
            except Exception:
                await retry_client.aclose()
                self.router.release(worker.url)
                raise

        t0 = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        needs_strip_vllm = self.strip_vllm
        needs_strip_logprobs = not originally_requested_logprobs

        async def event_generator():
            try:
                async for line in resp.aiter_lines():
                    # Parse SSE data lines for trace capture and sanitization
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            yield "data: [DONE]\n\n"
                            continue
                        try:
                            chunk = json.loads(data_str)
                            chunks.append(chunk)
                            if not needs_strip_vllm and not needs_strip_logprobs:
                                yield f"data: {data_str}\n\n"
                            else:
                                sanitized = strip_vllm_fields(chunk) if needs_strip_vllm else chunk
                                if needs_strip_logprobs:
                                    sanitized = _strip_logprobs(sanitized)
                                yield f"data: {json.dumps(sanitized)}\n\n"
                            continue
                        except json.JSONDecodeError:
                            pass
                    # Skip blank lines — SSE separators are already included
                    # in the \n\n suffix above
                    if not line:
                        continue
                    yield line + "\n"
            finally:
                await upstream.__aexit__(None, None, None)
                if retry_client is not None:
                    await retry_client.aclose()
                self.router.release(worker.url)

                latency_ms = (time.perf_counter() - t0) * 1000
                # Build trace from accumulated chunks.
                # NOTE: We use create_task instead of await because this
                # finally block may run during GeneratorExit, where await
                # on real async I/O (e.g. aiosqlite) is not reliable.
                if session_id and chunks:
                    trace = build_trace_record_from_chunks(session_id, request_body, chunks, latency_ms, weight_version=request.state.weight_version, capture_raw=self.capture_raw_payloads)
                    task = asyncio.create_task(
                        self._safe_store(
                            trace.trace_id,
                            trace.session_id,
                            trace.model_dump(),
                        )
                    )
                    self._pending_traces.add(task)
                    task.add_done_callback(self._pending_traces.discard)

                    # Ingest first turn into accumulator for cumulative token mode
                    if self.cumulative_token_mode:
                        acc = self._get_accumulator(session_id)
                        if acc.turn_count == 0:
                            prompt_ids = trace.prompt_token_ids
                            completion_ids = trace.completion_token_ids
                            if prompt_ids or completion_ids:
                                acc.ingest_turn(prompt_ids, completion_ids)
                                acc.update_prefix(request_body.get("messages", []))

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            status_code=resp.status_code,
        )

    async def _handle_streaming_local(
        self,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool = False,
        weight_version: int | None = None,
    ) -> StreamingResponse:
        """Handle streaming when using a local handler (fake-streaming)."""
        assert self.local_handler is not None
        t0 = time.perf_counter()
        response_body = await self.local_handler(request_body)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist trace from the full response
        if session_id and response_body:
            trace = build_trace_record(session_id, request_body, response_body, latency_ms, weight_version=weight_version, capture_raw=self.capture_raw_payloads)
            await self._persist(trace)

        needs_strip_vllm = self.strip_vllm
        needs_strip_logprobs = not originally_requested_logprobs

        # Build SSE chunks from the complete response
        chat_id = response_body.get("id", "chatcmpl-local")
        created = response_body.get("created", int(time.time()))
        model = response_body.get("model", "")
        choices = response_body.get("choices", [])
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {})
        finish_reason = first_choice.get("finish_reason", "stop")

        def _sanitize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
            sanitized = strip_vllm_fields(chunk) if needs_strip_vllm else chunk
            if needs_strip_logprobs:
                sanitized = _strip_logprobs(sanitized)
            return sanitized

        async def event_generator():
            def _sse(data: str) -> str:
                return f"data: {data}\n\n"

            # Chunk 1: role
            yield _sse(
                json.dumps(
                    _sanitize_chunk(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                        }
                    )
                )
            )

            # Chunk 2: full content + token data
            delta: dict[str, Any] = {}
            if message.get("content"):
                delta["content"] = message["content"]
            if message.get("reasoning"):
                delta["reasoning"] = message["reasoning"]
            if message.get("tool_calls"):
                delta["tool_calls"] = message["tool_calls"]

            content_chunk: dict[str, Any] = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                        "token_ids": first_choice.get("token_ids", []),
                        "logprobs": first_choice.get("logprobs"),
                    }
                ],
                "prompt_token_ids": response_body.get("prompt_token_ids", []),
            }
            yield _sse(json.dumps(_sanitize_chunk(content_chunk)))

            # Chunk 3: finish + usage
            yield _sse(
                json.dumps(
                    _sanitize_chunk(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                            "usage": response_body.get("usage", {}),
                        }
                    )
                )
            )

            yield _sse("[DONE]")

        return StreamingResponse(event_generator(), media_type="text/event-stream", status_code=200)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _send_with_abort_resume(
        self,
        *,
        method: str,
        url: str,
        content: bytes,
        headers: dict[str, str],
        request_body: dict[str, Any],
        worker_url: str,
        chat_response: bool,
    ) -> tuple[dict[str, Any], int]:
        """Resume structured vLLM aborts with the returned partial token IDs."""
        resp = await self._send_with_retry(method=method, url=url, content=content, headers=headers)
        status_code = resp.status_code
        try:
            response_body = json.loads(resp.content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}, status_code

        if not self.cumulative_token_mode or status_code != 200 or _finish_reason(response_body) != "abort":
            return response_body, status_code

        prompt_token_ids = extract_prompt_token_ids(response_body)
        completion_token_ids = extract_completion_token_ids(response_body)
        if not prompt_token_ids:
            logger.warning(
                "%svLLM returned finish_reason=abort without prompt token IDs",
                self.worker_label,
            )
            return response_body, status_code

        resume_body = {
            key: value
            for key, value in request_body.items()
            if key
            not in {
                "messages",
                "tools",
                "tool_choice",
                "stream_options",
                "chat_template",
                "add_generation_prompt",
                "continue_final_message",
            }
        }
        resume_body["stream"] = False
        resume_body["add_special_tokens"] = False
        requested_max_tokens = request_body.get("max_tokens")
        completions_url = self._build_url(worker_url, "/v1/completions", "")

        attempt = 0
        while _finish_reason(response_body) == "abort" and attempt < _MAX_ABORT_RESUMES:
            attempt += 1
            resume_body["prompt"] = prompt_token_ids + completion_token_ids
            if isinstance(requested_max_tokens, int):
                remaining = requested_max_tokens - len(completion_token_ids)
                if remaining <= 0:
                    response_body["choices"][0]["finish_reason"] = "length"
                    break
                resume_body["max_tokens"] = remaining
                self._clamp_max_tokens(resume_body, len(resume_body["prompt"]))

            logger.info(
                "%sresuming aborted vLLM generation (attempt %d, partial_tokens=%d)",
                self.worker_label,
                attempt,
                len(completion_token_ids),
            )
            await asyncio.sleep(1)
            resumed_resp = await self._send_with_retry(
                method="POST",
                url=completions_url,
                content=fastjson.dumps(resume_body),
                headers=headers,
            )
            status_code = resumed_resp.status_code
            try:
                resumed_body = json.loads(resumed_resp.content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}, status_code
            if status_code != 200:
                return resumed_body, status_code

            response_body = _merge_resumed_response(
                response_body,
                resumed_body,
                chat_response=chat_response,
                prompt_token_ids=prompt_token_ids,
            )
            completion_token_ids = extract_completion_token_ids(response_body)

        if _finish_reason(response_body) == "abort":
            logger.warning(
                "%sgeneration still aborted after %d resumes (partial_tokens=%d); returning partial turn",
                self.worker_label,
                attempt,
                len(completion_token_ids),
            )

        return response_body, status_code

    async def _send_with_retry(
        self,
        method: str,
        url: str,
        content: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        assert self._http is not None
        last_exc: Exception | None = None
        for attempt in range(1 + self.max_retries):
            retry_client: httpx.AsyncClient | None = None
            client = self._http
            if attempt > 0:
                retry_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout=None),
                    limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                    follow_redirects=True,
                )
                client = retry_client
            try:
                resp = await client.request(method, url, content=content, headers=headers)
                return resp
            except _RETRYABLE_HTTP_ERRORS as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Transient HTTP error (attempt %d/%d, type=%s): %s",
                        attempt + 1,
                        self.max_retries + 1,
                        type(exc).__name__,
                        exc,
                    )
            finally:
                if retry_client is not None:
                    await retry_client.aclose()
        raise last_exc  # type: ignore[misc]

    async def _persist(self, trace: TraceRecord) -> None:
        try:
            data = trace.model_dump()
            if self.sync_traces:
                await self.store.store_trace(trace.trace_id, trace.session_id, data)
            else:
                task = asyncio.create_task(self._safe_store(trace.trace_id, trace.session_id, data))
                self._pending_traces.add(task)
                task.add_done_callback(self._pending_traces.discard)
        except Exception:
            logger.exception("Failed to persist trace %s", trace.trace_id)

    async def _safe_store(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        try:
            await self.store.store_trace(trace_id, session_id, data)
        except Exception:
            logger.exception("Failed to persist trace %s", trace_id)

    async def _persist_trace(
        self,
        session_id: str,
        request_body: dict[str, Any],
        response_body: dict[str, Any],
        latency_ms: float,
        weight_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Build + store a trace off the event-loop thread and off the response
        critical path.

        ``build_trace_record`` + ``model_dump`` copy the ≤120K/16K token-id lists
        and were previously done inline on the loop before returning the response.
        Here they run in the executor (``_build_trace_data``); only the async store
        write touches the loop, and the response no longer waits for any of it.
        ``sync_traces`` still forces synchronous completion for callers that need it.
        """
        loop = asyncio.get_running_loop()
        capture_raw = self.capture_raw_payloads

        async def _run() -> None:
            try:
                trace_id, sess, data = await loop.run_in_executor(
                    None,
                    _build_trace_data,
                    session_id,
                    request_body,
                    response_body,
                    latency_ms,
                    weight_version,
                    capture_raw,
                    metadata,
                )
                await self._safe_store(trace_id, sess, data)
            except Exception:
                logger.exception("Failed to persist trace (session=%s)", session_id)

        if self.sync_traces:
            await _run()
        else:
            task = asyncio.create_task(_run())
            self._pending_traces.add(task)
            task.add_done_callback(self._pending_traces.discard)

    @staticmethod
    def _build_url(worker_url: str, path: str, query: str, *, gateway_prefix: str = "/v1") -> str:
        base = worker_url.rstrip("/")
        # Strip the gateway's own prefix to get the tail (e.g. /chat/completions).
        # The gateway always exposes routes under /v1/{path}, so request paths
        # arrive as /v1/... regardless of the worker's actual api_path.
        if path.startswith(gateway_prefix):
            path = path[len(gateway_prefix) :]
        url = f"{base}{path}"
        if query:
            url = f"{url}?{query}"
        return url

    @staticmethod
    def _forward_headers(request: Request, session_id: str | None = None) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        # Pin a trajectory's turns to one replica so Fireworks reuses its
        # prompt-prefix KV (caching is per-replica). The training path sets these
        # via the rollout engine; the HTTP path (eval) does it here. No-op for
        # non-Fireworks upstreams; setdefault lets a client pin its own value.
        if session_id:
            headers.setdefault("x-session-affinity", str(session_id))
            headers.setdefault("x-multi-turn-session-id", str(session_id))
        return headers
