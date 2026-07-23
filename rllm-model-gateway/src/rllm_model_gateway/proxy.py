"""httpx-based reverse proxy with streaming SSE support.

Reference: miles ``MilesRouter._do_proxy()``
(``miles/router/router.py`` lines 138-166).
"""

import asyncio
import functools
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from rllm_model_gateway import fastjson
from rllm_model_gateway.data_process import (
    build_trace_record,
    build_trace_record_from_chunks,
    extract_completion_token_ids,
    extract_prompt_token_ids,
    strip_vllm_fields,
)
from rllm_model_gateway.models import TraceRecord
from rllm_model_gateway.session_router import SessionRouter
from rllm_model_gateway.store.base import TraceStore
from rllm_model_gateway.token_accumulator import (
    SessionSlots,
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


def _get_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _to_openai_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Renderer ``parse_response`` tool_calls -> OpenAI chat ``tool_calls`` shape.

    The renderer emits ``{"function": {"name", "arguments": <dict|str>}}``; the OpenAI
    wire shape a client (e.g. opencode's OpenAI-compatible provider) expects is
    ``{"id", "type": "function", "index", "function": {"name", "arguments": <json str>}}``.
    """
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        fn = _get_field(tc, "function")
        if fn is not None:
            name = _get_field(fn, "name", "")
            args = _get_field(fn, "arguments", {})
        else:
            name = _get_field(tc, "name", "")
            args = _get_field(tc, "arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args)
        out.append(
            {
                "id": _get_field(tc, "id") or f"call_{i}",
                "type": "function",
                "index": i,
                "function": {"name": name, "arguments": args},
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
    lineage_id: str | None = None,
    trace_id: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Build a TraceRecord and serialize it to a dict.

    Runs in a worker thread (see ``ReverseProxy._persist_trace``) so the token-id
    list copies + ``model_dump`` stay off the event-loop thread. Reads
    ``request_body``/``response_body`` without mutating them, so it's safe to run
    concurrently with the response path. ``lineage_id`` is resolved on the loop
    (the active slot) before dispatch, so this stays side-effect-free.
    """
    trace = build_trace_record(
        session_id,
        request_body,
        response_body,
        latency_ms,
        weight_version=weight_version,
        lineage_id=lineage_id,
        trace_id=trace_id,
        capture_raw=capture_raw,
    )
    return trace.trace_id, trace.session_id, trace.model_dump()


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
        heartbeat_initial_delay_s: float = 50.0,
        heartbeat_interval_s: float = 25.0,
        heartbeat_budget_s: float = 3600.0,
        capture_raw_payloads: bool = False,
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
        # Retain full raw request/response on each trace. Off by default: training
        # reads only token-id/logprob/message fields, and serializing the raw
        # dicts (≤120K-token prompt + full response) is the dominant per-request
        # CPU cost on the event loop at high concurrency. Enable for debugging.
        self.capture_raw_payloads = capture_raw_payloads
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
        # One SessionSlots per session, holding an accumulator per conversation
        # lineage (parent agent + any subagents) so interleaved lineages each
        # stay bridged (drift-free) instead of forcing resets on each switch.
        self._accumulators: dict[str, SessionSlots] = {}
        # Loop-health instrumentation (diagnostic only; see _loop_health_monitor).
        # _inflight counts concurrent in-flight generations; _recent_lag_ms is the
        # latest event-loop lag sample, stamped onto duplicate logs for correlation.
        self._inflight: int = 0
        self._inflight_max: int = 0
        self._recent_lag_ms: float = 0.0
        self._monitor_task: asyncio.Task[None] | None = None

    def _session_slots(self, session_id: str) -> SessionSlots:
        """Return the SessionSlots registry for *session_id*, creating if needed."""
        slots = self._accumulators.get(session_id)
        if slots is None:
            slots = SessionSlots(self.renderer, session_id=session_id)
            self._accumulators[session_id] = slots
        return slots

    def _request_slot(self, request: Request) -> Any:
        """The chain (slot) ``select()``/``fork()`` bound to THIS request, or None.

        Read this — never ``SessionSlots.active`` — in the post-generation
        ingest/tag. ``active`` is a shared per-session pointer that a concurrent
        same-session request (e.g. opencode's async title-generation call) can
        clobber between ``select()`` and here, appending/tagging a trace onto the
        wrong lineage. ``request.state.slot`` is fixed for the request's lifetime,
        so the lineage is a pure function of the chain this request extends.
        ``None`` when cumulative mode is off / no slot was selected.
        """
        return getattr(request.state, "slot", None)

    @staticmethod
    def _slot_lineage_id(slot: Any) -> str | None:
        return slot.lineage_id if slot is not None else None

    def _request_lineage_id(self, request: Request) -> str | None:
        return self._slot_lineage_id(self._request_slot(request))

    @staticmethod
    def _ingest_turn0(slot: Any, prompt_ids: list[int], completion_ids: list[int], messages: list[dict[str, Any]]) -> None:
        """Seed a freshly selected/forked chain with its turn-0 tokens.

        No-op when the slot is absent, already advanced past turn 0, or carries no
        tokens. Bound to the request's own chain (via ``_request_slot``), so a
        concurrent request touching ``.active`` can't misdirect the ingest.
        """
        if slot is not None and slot.turn_count == 0 and (prompt_ids or completion_ids):
            slot.ingest_turn(prompt_ids, completion_ids)
            slot.update_prefix(messages)

    @staticmethod
    def _turn_trace_id(slot: Any, replay: bool) -> str:
        """Trace id for this turn's persisted trace.

        Delegates to the slot so a replay reuses the turn's existing id — the
        store upserts by trace_id, so it overwrites that turn's trace in place
        instead of appending a second, superseded trace. Falls back to a fresh id
        when there is no slot (cumulative mode off / non-chat request).
        """
        if slot is None:
            return str(uuid.uuid4())
        return slot.next_trace_id(replay)

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
        if self._monitor_task is None:
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
            messages = request_body.get("messages", [])
            # Route to the chain whose lineage this request continues (a new chain
            # if it continues none — e.g. a subagent / title-generation turn).
            acc = self._session_slots(session_id).select(messages)
            # Bind THIS request to the chain select() chose, so its turn-0 ingest
            # and lineage tag use this chain — not the shared .active pointer a
            # concurrent same-session request could clobber between here and the
            # post-generation ingest/tag.
            request.state.slot = acc
            if acc.should_rewrite():
                # Classify the incoming request against the matched chain.
                # plan.action is "extend" (build a /v1/completions bridge),
                # "replay" (duplicate resend — regenerate in place), or a reset
                # reason (empty-delta / renderer-can't-bridge) which we handle by
                # FORKING a fresh chain and re-ingesting this turn as its turn-0 —
                # the matched chain stays immutable (one lineage == one token chain).
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
                    # Fork a fresh chain and re-ingest this turn as its turn-0,
                    # rather than resetting the matched chain in place — chains stay
                    # immutable (one lineage == one token chain), so a bridge gap
                    # never splits the matched lineage into two token segments.
                    logger.info(
                        "%sTokenAccumulator renderer_no_bridge session=%s turn=%d renderer=%s new_roles=%s: forking new lineage",
                        self.worker_label,
                        session_id,
                        acc.turn_count,
                        type(acc.renderer).__name__,
                        [m.get("role") for m in plan.new_messages],
                    )
                    acc = self._session_slots(session_id).fork()
                    request.state.slot = acc
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
                    # Matched chain but nothing renderable to bridge (empty delta):
                    # fork a fresh chain instead of resetting the matched one.
                    logger.info(
                        "%sTokenAccumulator %s session=%s turn=%d: forking new lineage",
                        self.worker_label,
                        plan.reason.value if plan.reason else "reset",
                        session_id,
                        acc.turn_count,
                    )
                    acc = self._session_slots(session_id).fork()
                    request.state.slot = acc

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
                resp = await self._send_with_retry(
                    method=request.method,
                    url=url,
                    content=raw_body,
                    headers=headers,
                )
                content = resp.content
                status_code = resp.status_code
            finally:
                self.router.release(worker.url)

            # Parse response for trace extraction
            try:
                response_body = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_body = {}

        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist trace
        if session_id and response_body:
            await self._persist_trace(
                session_id, request_body, response_body, latency_ms, request.state.weight_version, self._request_lineage_id(request), self._turn_trace_id(self._request_slot(request), replay=False)
            )

            # Ingest turn-0 into the chain this request was bound to (never .active).
            if self.cumulative_token_mode and request.url.path.endswith("/chat/completions"):
                self._ingest_turn0(
                    self._request_slot(request),
                    extract_prompt_token_ids(response_body),
                    extract_completion_token_ids(response_body),
                    request_body.get("messages", []),
                )

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

        ``token_ids`` is the full bridge-extended prompt for this turn, built by
        ``acc.build_next_prompt`` in ``handle()`` — or, when ``replay`` is set,
        the prior turn's prompt being regenerated after a duplicate resend.
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
                resp = await self._send_with_retry(
                    method="POST",
                    url=url,
                    content=raw_body,
                    headers=headers,
                )
                content = resp.content
                status_code = resp.status_code
            finally:
                self.router.release(worker.url)

            try:
                response_body = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_body = {}

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
            await self._persist_trace(session_id, request_body, response_body, latency_ms, request.state.weight_version, self._request_lineage_id(request), self._turn_trace_id(acc, replay))

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
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as first_exc:
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

        stream_trace_id = self._turn_trace_id(acc, replay)

        def _build_trace():
            latency_ms = (time.perf_counter() - t0) * 1000
            return build_trace_record_from_chunks(
                session_id,
                request_body,
                chunks,
                latency_ms,
                weight_version=request.state.weight_version,
                lineage_id=self._request_lineage_id(request),
                trace_id=stream_trace_id,
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
                weight_version=request.state.weight_version,
                lineage_id=self._request_lineage_id(request),
                trace_id=self._turn_trace_id(acc, replay),
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
            return await self._handle_streaming_local(request_body, session_id, originally_requested_logprobs, request.state.weight_version, slot=self._request_slot(request))

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
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as first_exc:
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
                    trace = build_trace_record_from_chunks(
                        session_id,
                        request_body,
                        chunks,
                        latency_ms,
                        weight_version=request.state.weight_version,
                        lineage_id=self._request_lineage_id(request),
                        trace_id=self._turn_trace_id(self._request_slot(request), replay=False),
                        capture_raw=self.capture_raw_payloads,
                    )
                    task = asyncio.create_task(
                        self._safe_store(
                            trace.trace_id,
                            trace.session_id,
                            trace.model_dump(),
                        )
                    )
                    self._pending_traces.add(task)
                    task.add_done_callback(self._pending_traces.discard)

                    # Ingest turn-0 into the chain this request was bound to.
                    if self.cumulative_token_mode:
                        self._ingest_turn0(
                            self._request_slot(request),
                            trace.prompt_token_ids,
                            trace.completion_token_ids,
                            request_body.get("messages", []),
                        )

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
        slot: Any = None,
    ) -> StreamingResponse:
        """Handle streaming when using a local handler (fake-streaming)."""
        assert self.local_handler is not None
        t0 = time.perf_counter()
        response_body = await self.local_handler(request_body)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist trace from the full response
        if session_id and response_body:
            trace = build_trace_record(
                session_id,
                request_body,
                response_body,
                latency_ms,
                weight_version=weight_version,
                lineage_id=self._slot_lineage_id(slot),
                trace_id=self._turn_trace_id(slot, replay=False),
                capture_raw=self.capture_raw_payloads,
            )
            await self._persist(trace)

            # Seed turn-0 into the chain this request was bound to (`slot`, chosen
            # by the caller). Without a turn-0 ingest the chain never advances, so
            # continues() stays False and every later turn forks a new lineage —
            # one trajectory per turn. Binding to `slot` (not .active) also stops a
            # concurrent same-session request from misdirecting the ingest.
            if self.cumulative_token_mode:
                self._ingest_turn0(slot, trace.prompt_token_ids, trace.completion_token_ids, request_body.get("messages", []))

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
            try:
                resp = await self._http.request(method, url, content=content, headers=headers)
                return resp
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Connection error (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
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
        lineage_id: str | None = None,
        trace_id: str | None = None,
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
                built_trace_id, sess, data = await loop.run_in_executor(
                    None,
                    _build_trace_data,
                    session_id,
                    request_body,
                    response_body,
                    latency_ms,
                    weight_version,
                    capture_raw,
                    lineage_id,
                    trace_id,
                )
                await self._safe_store(built_trace_id, sess, data)
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
