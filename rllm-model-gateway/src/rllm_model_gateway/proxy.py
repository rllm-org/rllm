"""httpx-based reverse proxy with streaming SSE support.

Reference: miles ``MilesRouter._do_proxy()``
(``miles/router/router.py`` lines 138-166).
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from rllm_model_gateway import sglang_helper
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
    TokenAccumulator,
    extract_new_messages,
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
        use_sglang: bool = False,
        renderer: Any = None,
        tokenizer: Any = None,
        model: str | None = None,
        sglang_tool_call_parser: str | None = None,
        sglang_reasoning_parser: str | None = None,
    ) -> None:
        self.router = router
        self.store = store
        self.strip_vllm = strip_vllm
        self.sync_traces = sync_traces
        self.max_retries = max_retries
        self.local_handler = local_handler
        self.cumulative_token_mode = cumulative_token_mode
        self.use_sglang = use_sglang
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.model = model
        self.sglang_tool_call_parser = sglang_tool_call_parser
        self.sglang_reasoning_parser = sglang_reasoning_parser
        # Lazily-built SGLang FunctionCallParser (depends on the request's tools).
        self._fc_parser_cache: dict[str, Any] = {}
        # Probe whether the tokenizer auto-prepends special tokens (e.g. BOS) on
        # encode(). SGLang's chat path splits apply_chat_template into render +
        # encode and passes add_special_tokens=False when this is True, so the
        # chat template's own role/special tokens aren't doubled (e.g. double BOS
        # on Llama/Kimi-style tokenizers). We mirror that to tokenize a message
        # list byte-identically to SGLang's /v1/chat/completions. Probed once.
        self._tokenizer_auto_adds_specials = True
        if self.tokenizer is not None:
            try:
                self._tokenizer_auto_adds_specials = len(self.tokenizer.encode("")) > 0
            except Exception:
                self._tokenizer_auto_adds_specials = True
        self.weight_version: int | None = None
        self._http: httpx.AsyncClient | None = None
        self._pending_traces: set[asyncio.Task[None]] = set()
        self._accumulators: dict[str, TokenAccumulator] = {}

    def _get_accumulator(self, session_id: str) -> TokenAccumulator:
        """Return the TokenAccumulator for *session_id*, creating if needed."""
        if session_id not in self._accumulators:
            self._accumulators[session_id] = TokenAccumulator(self.renderer)
        return self._accumulators[session_id]

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=None),  # no timeout — LLM calls can be long
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
            follow_redirects=True,
        )

    async def stop(self) -> None:
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
            request_body = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_body = {}

        is_stream = request_body.get("stream", False)

        # use_sglang: route every chat turn through SGLang's native /generate API.
        # The gateway renders the prompt to token ids itself (full render on turn 0
        # or after a prefix break; cross-turn bridge otherwise), so /generate
        # returns token ids + logprobs in meta_info natively — no sglang patches for
        # RL, and it works through sgl-router (which rejects token-id prompts on
        # /v1/completions).
        if self.use_sglang and session_id and request.url.path.endswith("/chat/completions"):
            return await self._handle_sglang_generate(request, request_body, session_id, is_stream, originally_requested_logprobs)

        # Cumulative token mode interception: if enabled and past first turn,
        # rewrite to /v1/completions with pre-tokenized prompt to avoid drift.
        if self.cumulative_token_mode and session_id and request.url.path.endswith("/chat/completions"):
            acc = self._get_accumulator(session_id)
            if acc.should_rewrite():
                messages = request_body.get("messages", [])
                if not acc.is_cumulative(messages):
                    # Message history diverged — reset and fall through to
                    # normal chat path (treated as fresh turn-0).
                    acc.reset()
                else:
                    new_messages = extract_new_messages(messages, acc.message_count)
                    token_ids = None
                    if new_messages:
                        token_ids = acc.build_next_prompt(new_messages, tools=request_body.get("tools"))
                    if token_ids is not None:
                        return await self._handle_cumulative_turn(
                            request,
                            request_body,
                            session_id,
                            acc,
                            token_ids,
                            originally_requested_logprobs,
                        )
                    # No new messages, or the renderer couldn't prove the
                    # prefix-extension contract (e.g. DefaultRenderer, or an
                    # assistant message in the new slice). Reset so this turn is
                    # re-ingested as a fresh turn-0 on the chat path; otherwise
                    # the stale prefix would drop this turn's completion tokens
                    # from the next cumulative prompt and break prefix-extension.
                    acc.reset()

        if is_stream:
            return await self._handle_streaming(request, body, request_body, session_id, originally_requested_logprobs)
        return await self._handle_non_streaming(request, body, request_body, session_id, originally_requested_logprobs)

    # ------------------------------------------------------------------
    # SGLang native /generate path (use_sglang)
    # ------------------------------------------------------------------

    def _render_sglang_prompt(self, session_id: str, request_body: dict[str, Any]) -> list[int]:
        """Render this chat turn's prompt to token ids for SGLang /generate.

        The renderer is used for the cross-turn BRIDGE ONLY. The bridge is a
        narrow interception — it applies just on turn>0 cumulative turns whose
        history cumulatively extends the tracked prefix, where it extends
        ``prev_prompt + prev_completion`` with only the new (non-assistant)
        messages, keeping sampled tokens verbatim so adjacent turns share a
        byte-exact prefix (mergeable). EVERYTHING ELSE — non-cumulative mode,
        cumulative turn 0, a prefix break, or a declined bridge — falls through to
        the common path: tokenize the full message list client-side with the
        model's HF chat template (authoritative; mirrors slime's _render_token_ids
        and works through sgl-router, which exposes no messages→token-ids endpoint).
        """
        messages = request_body.get("messages", []) or []
        tools = request_body.get("tools")

        # Cumulative bridge: the only path that uses the renderer.
        if self.cumulative_token_mode:
            acc = self._get_accumulator(session_id)
            if acc.should_rewrite() and acc.is_cumulative(messages):
                new_messages = extract_new_messages(messages, acc.message_count)
                if new_messages:
                    # Flatten block content before the renderer (it expects str).
                    token_ids = acc.build_next_prompt([sglang_helper.flatten_message_content(m) for m in new_messages], tools=tools)
                    if token_ids is not None:
                        return token_ids
            # Bridge did not apply (turn 0, divergent/declined): reset prefix
            # bookkeeping and fall through to the common full-render path.
            acc.reset()

        # Common path (non-cumulative always; cumulative turn-0/reset): full render
        # via the model's HF chat template (delegated to sglang_helper, which
        # mirrors SGLang's ServingChat._apply_jinja_template exactly).
        return sglang_helper.apply_chat_template(
            messages,
            tools,
            tokenizer=self.tokenizer,
            auto_adds_specials=self._tokenizer_auto_adds_specials,
        )

    def _parse_completion(self, completion_ids: list[int], tools: Any) -> dict[str, Any]:
        """Parse /generate completion ids into ``{content, reasoning, tool_calls}``
        with SGLang's parsers (thin wrapper over sglang_helper.parse_completion,
        binding this proxy's tokenizer / parser names / parser cache)."""
        return sglang_helper.parse_completion(
            completion_ids,
            tools,
            tokenizer=self.tokenizer,
            tool_call_parser=self.sglang_tool_call_parser,
            reasoning_parser=self.sglang_reasoning_parser,
            fc_cache=self._fc_parser_cache,
        )

    async def _handle_sglang_generate(
        self,
        request: Request,
        request_body: dict[str, Any],
        session_id: str,
        is_stream: bool,
        originally_requested_logprobs: bool = False,
    ) -> Response:
        """Run one chat turn via SGLang's native /generate (token-in, token-out).

        Renders the prompt to token ids, posts to <worker>/generate with
        return_logprob=True, captures token ids + logprobs from meta_info into a
        TraceRecord, updates the accumulator, and returns an OpenAI-chat-shaped
        response to the agent (so the client sees a normal chat completion).
        """
        t0 = time.perf_counter()

        # Tokenize the prompt client-side (model chat template / cumulative bridge).
        prompt_ids = self._render_sglang_prompt(session_id, request_body)

        worker = self.router.route(session_id)
        headers = self._forward_headers(request)
        # Keep a session sticky to one engine (radix/prefix-cache affinity).
        headers["X-SMG-Routing-Key"] = session_id

        gen_body: dict[str, Any] = {
            "rid": uuid.uuid4().hex,
            "input_ids": prompt_ids,
            "sampling_params": sglang_helper.sglang_sampling_params(request_body),
            "return_logprob": True,
            "stream": bool(is_stream),
        }
        # /generate lives at the worker root, not under the OpenAI /v1 prefix.
        url = f"{worker.url.rstrip('/')}/generate"
        raw_body = json.dumps(gen_body).encode()

        assert self._http is not None
        if is_stream:
            # The streaming generator owns the worker release (after [DONE]); do
            # not release here or we'd double-decrement the active count.
            return await self._sglang_generate_streaming(request, request_body, session_id, prompt_ids, url, headers, raw_body, t0)
        try:
            resp = await self._http.post(url, content=raw_body, headers=headers)
        finally:
            self.router.release(worker.url)

        content = resp.content
        status_code = resp.status_code
        if status_code >= 400:
            logger.warning("sglang /generate %s: %.300s", status_code, content[:300])
            return Response(content=content, status_code=status_code, media_type="application/json")

        try:
            gen = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            gen = {}
        if isinstance(gen, list):  # n>1 returns a list; we use n=1
            gen = gen[0] if gen else {}

        prompt_ids_out, completion_ids, logprobs, text, finish_reason = sglang_helper.parse_sglang_generate(gen, prompt_ids)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Update accumulator bookkeeping so the next turn can bridge — only needed
        # in cumulative mode (non-cumulative full-renders every turn statelessly).
        if self.cumulative_token_mode:
            acc = self._get_accumulator(session_id)
            acc.ingest_turn(prompt_ids_out, completion_ids)
            acc.update_prefix(request_body.get("messages", []))

        trace = sglang_helper.build_generate_trace(
            session_id,
            request_body,
            prompt_ids_out,
            completion_ids,
            logprobs,
            text,
            finish_reason,
            latency_ms,
            weight_version=request.state.weight_version,
        )
        await self._persist(trace)

        # Parse the completion token ids back into structured tool_calls so the
        # agent (which reads message.tool_calls, not raw text) can act on them —
        # /generate returns only raw text, so we parse with SGLang's own parser
        # (same as /v1/chat/completions).
        parsed = self._parse_completion(completion_ids, request_body.get("tools"))
        chat = sglang_helper.generate_to_chat_response(request_body, text, finish_reason, len(prompt_ids_out), len(completion_ids), parsed=parsed)
        return Response(content=json.dumps(chat), status_code=200, media_type="application/json")

    async def _sglang_generate_streaming(
        self,
        request: Request,
        request_body: dict[str, Any],
        session_id: str,
        prompt_ids: list[int],
        url: str,
        headers: dict[str, str],
        raw_body: bytes,
        t0: float,
    ) -> StreamingResponse:
        """Stream SGLang /generate, translate to OpenAI chat SSE, capture trace.

        SGLang /generate has no token-level tool-call parsing, so we cannot stream
        raw text deltas as ``content`` (a ``<tool_call>`` would reach the agent as
        text, not structured tool_calls). Following slime's adapter, we realise the
        whole turn server-side (consume the full SGLang stream), parse it, then emit
        the OpenAI chat SSE as single role/content/tool_calls/finish chunks.
        """
        worker_url_for_release = url.rsplit("/generate", 1)[0]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = request_body.get("model", "")

        def _chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n".encode()

        async def event_generator():
            last: dict[str, Any] = {}
            assert self._http is not None
            try:
                async with self._http.stream("POST", url, content=raw_body, headers=headers) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        logger.warning("sglang /generate stream %s: %.300s", resp.status_code, body[:300])
                        yield b"data: [DONE]\n\n"
                        return
                    # Consume the full (cumulative) stream; keep the last payload.
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            last = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
            finally:
                self.router.release(worker_url_for_release)

            # Realise + parse the whole turn, then emit slime-style SSE.
            p_ids, c_ids, lps, text, finish = sglang_helper.parse_sglang_generate(last, prompt_ids)
            parsed = self._parse_completion(c_ids, request_body.get("tools"))
            tool_calls = sglang_helper.to_openai_tool_calls(parsed.get("tool_calls") or [])
            content = parsed.get("content")
            reasoning = parsed.get("reasoning")

            # Emit content + tool_calls + reasoning losslessly (mirrors SGLang
            # native): the leftover text is always sent, never dropped because a
            # tool call is present, so the agent can replay the full message next
            # turn and we re-tokenize it identically.
            wire_finish = finish or "stop"
            yield _chunk({"role": "assistant"})
            if reasoning:
                yield _chunk({"reasoning_content": reasoning})
            body_text = content if content is not None else text
            if body_text:
                yield _chunk({"content": body_text})
            if tool_calls:
                yield _chunk({"tool_calls": [{**tc, "index": i} for i, tc in enumerate(tool_calls)]})
                wire_finish = "tool_calls"
            yield _chunk({}, finish=wire_finish)
            yield b"data: [DONE]\n\n"

            if self.cumulative_token_mode:
                acc = self._get_accumulator(session_id)
                acc.ingest_turn(p_ids, c_ids)
                acc.update_prefix(request_body.get("messages", []))
            trace = sglang_helper.build_generate_trace(
                session_id,
                request_body,
                p_ids,
                c_ids,
                lps,
                text,
                finish,
                (time.perf_counter() - t0) * 1000,
                weight_version=request.state.weight_version,
            )
            task = asyncio.create_task(self._safe_store(trace.trace_id, trace.session_id, trace.model_dump()))
            self._pending_traces.add(task)
            task.add_done_callback(self._pending_traces.discard)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

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
        t0 = time.perf_counter()

        if self.local_handler is not None:
            # In-process path: call handler directly, no HTTP
            response_body = await self.local_handler(request_body)
            status_code = 200
        else:
            # HTTP proxy path
            worker = self.router.route(session_id)
            url = self._build_url(worker.api_url, request.url.path, str(request.url.query))
            headers = self._forward_headers(request)
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
            trace = build_trace_record(session_id, request_body, response_body, latency_ms, weight_version=request.state.weight_version)
            await self._persist(trace)

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

        return Response(
            content=json.dumps(sanitized),
            status_code=status_code,
            media_type="application/json",
        )

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
    ) -> Response:
        """Rewrite chat/completions to /v1/completions with pre-tokenized prompt.

        ``token_ids`` is the full bridge-extended prompt for this turn, built
        by ``acc.build_next_prompt`` in ``handle()``.

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
            return await self._handle_cumulative_streaming(request, request_body, completions_body, session_id, acc, token_ids)
        return await self._handle_cumulative_non_streaming(
            request,
            request_body,
            completions_body,
            session_id,
            acc,
            token_ids,
            originally_requested_logprobs,
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
    ) -> Response:
        """Non-streaming cumulative turn: send non-streaming to vLLM, return JSON."""
        t0 = time.perf_counter()

        worker = self.router.route(session_id)
        url = self._build_url(worker.api_url, "/v1/completions", "")
        headers = self._forward_headers(request)
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

        acc.ingest_turn(prompt_token_ids, completion_token_ids)
        acc.update_prefix(request_body.get("messages", []))

        # Translate to chat format
        choices = response_body.get("choices") or []
        if choices:
            first_choice = choices[0]
            first_choice["message"] = {"role": "assistant", "content": first_choice.pop("text", "")}
        response_body["object"] = "chat.completion"

        if session_id and response_body:
            trace = build_trace_record(session_id, request_body, response_body, latency_ms, weight_version=request.state.weight_version)
            await self._persist(trace)

        sanitized = response_body
        if isinstance(response_body, dict) and response_body:
            if self.strip_vllm:
                sanitized = strip_vllm_fields(response_body)
            if not originally_requested_logprobs:
                sanitized = _strip_logprobs(sanitized)

        return Response(
            content=json.dumps(sanitized),
            status_code=status_code,
            media_type="application/json",
        )

    async def _handle_cumulative_streaming(
        self,
        request: Request,
        request_body: dict[str, Any],
        completions_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
    ) -> StreamingResponse:
        """Streaming cumulative turn: stream from vLLM, translate chunks to chat format."""
        completions_body["stream"] = True

        worker = self.router.route(session_id)
        url = self._build_url(worker.api_url, "/v1/completions", "")
        headers = self._forward_headers(request)
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

        async def event_generator():
            try:
                first_chunk_sent = False
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        if line:
                            yield line + "\n"
                        continue

                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield "data: [DONE]\n\n"
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    chunks.append(chunk)

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

            finally:
                await upstream.__aexit__(None, None, None)
                if retry_client is not None:
                    await retry_client.aclose()
                self.router.release(worker.url)

                # Ingest accumulated token data
                if chunks:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    trace = build_trace_record_from_chunks(session_id, request_body, chunks, latency_ms, weight_version=request.state.weight_version)
                    prompt_ids = trace.prompt_token_ids or token_ids
                    completion_ids = trace.completion_token_ids

                    acc.ingest_turn(prompt_ids, completion_ids)
                    acc.update_prefix(request_body.get("messages", []))

                    task = asyncio.create_task(self._safe_store(trace.trace_id, trace.session_id, trace.model_dump()))
                    self._pending_traces.add(task)
                    task.add_done_callback(self._pending_traces.discard)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            status_code=resp.status_code,
        )

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
        headers = self._forward_headers(request)

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
                    trace = build_trace_record_from_chunks(session_id, request_body, chunks, latency_ms, weight_version=request.state.weight_version)
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
            trace = build_trace_record(session_id, request_body, response_body, latency_ms, weight_version=weight_version)
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
    def _forward_headers(request: Request) -> dict[str, str]:
        return {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
