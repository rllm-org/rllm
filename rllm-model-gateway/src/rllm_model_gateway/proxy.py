"""httpx-based reverse proxy with streaming SSE support.

Reference: miles ``MilesRouter._do_proxy()``
(``miles/router/router.py`` lines 138-166).
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from rllm_model_gateway.data_process import (
    build_trace_record,
    build_trace_record_from_chunks,
    strip_vllm_fields,
)
from rllm_model_gateway.models import TraceRecord
from rllm_model_gateway.session_router import SessionRouter
from rllm_model_gateway.store.base import TraceStore

logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
_CONTEXT_BUDGET_METADATA_KEY = "rllm_gateway_context_budget"
_MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
_CHAT_TOKENIZE_FIELDS = frozenset(
    {
        "model",
        "messages",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
        "chat_template",
        "chat_template_kwargs",
        "mm_processor_kwargs",
        "tools",
    }
)

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


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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
        max_prompt_length: int | None = None,
    ) -> None:
        self.router = router
        self.store = store
        self.strip_vllm = strip_vllm
        self.sync_traces = sync_traces
        self.max_retries = max_retries
        self.local_handler = local_handler
        self.max_prompt_length = max_prompt_length
        self._http: httpx.AsyncClient | None = None
        self._pending_traces: set[asyncio.Task[None]] = set()

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

    async def _prepare_chat_context_budget(
        self,
        *,
        worker_url: str,
        path: str,
        request_body: dict[str, Any],
        raw_body: bytes,
        headers: dict[str, str],
    ) -> tuple[bytes, dict[str, Any], dict[str, Any], dict[str, Any] | None, int | None]:
        """Adjust chat ``max_tokens`` to fit model context without truncating prompts."""
        if not path.endswith(_CHAT_COMPLETIONS_SUFFIX):
            return raw_body, request_body, {}, None, None

        max_tokens_key = "max_completion_tokens" if request_body.get("max_completion_tokens") is not None else "max_tokens"
        requested_max_tokens = _as_positive_int(request_body.get(max_tokens_key))
        if requested_max_tokens is None:
            return raw_body, request_body, {}, None, None

        tokenize_body = {key: request_body[key] for key in _CHAT_TOKENIZE_FIELDS if key in request_body}
        if "messages" not in tokenize_body:
            return raw_body, request_body, {}, None, None

        tokenize_url = f"{worker_url.rstrip('/')}/tokenize"
        try:
            tokenize_resp = await self._send_with_retry(
                method="POST",
                url=tokenize_url,
                content=json.dumps(tokenize_body).encode("utf-8"),
                headers=headers,
            )
            tokenize_body_resp = tokenize_resp.json()
        except Exception as exc:
            logger.warning("Failed to tokenize chat request for context budgeting: %s", exc)
            return raw_body, request_body, {}, None, None

        if tokenize_resp.status_code >= 400:
            logger.warning(
                "Skipping context budgeting because %s returned status %s: %s",
                tokenize_url,
                tokenize_resp.status_code,
                tokenize_body_resp,
            )
            return raw_body, request_body, {}, None, None

        prompt_tokens = _as_positive_int(tokenize_body_resp.get("count"))
        max_model_len = _as_positive_int(tokenize_body_resp.get("max_model_len"))
        if prompt_tokens is None or max_model_len is None:
            return raw_body, request_body, {}, None, None

        remaining = max_model_len - prompt_tokens
        context_metadata: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "max_model_len": max_model_len,
            "requested_max_tokens": requested_max_tokens,
            "remaining_tokens": remaining,
            "max_tokens_key": max_tokens_key,
        }

        if self.max_prompt_length and prompt_tokens > self.max_prompt_length:
            context_metadata["termination_reason"] = _MAX_PROMPT_LENGTH_EXCEEDED
            body = self._build_context_error_body(
                request_body=request_body,
                tokenized=tokenize_body_resp,
                metadata=context_metadata,
            )
            return raw_body, request_body, {_CONTEXT_BUDGET_METADATA_KEY: context_metadata}, body, 400

        if remaining <= 0:
            context_metadata["termination_reason"] = _MAX_PROMPT_LENGTH_EXCEEDED
            body = self._build_context_error_body(
                request_body=request_body,
                tokenized=tokenize_body_resp,
                metadata=context_metadata,
            )
            return raw_body, request_body, {_CONTEXT_BUDGET_METADATA_KEY: context_metadata}, body, 400

        if remaining >= requested_max_tokens:
            return raw_body, request_body, {_CONTEXT_BUDGET_METADATA_KEY: context_metadata}, None, None

        adjusted_body = dict(request_body)
        adjusted_body[max_tokens_key] = remaining
        context_metadata.update(
            {
                "max_tokens_adjusted": True,
                "adjusted_max_tokens": remaining,
            }
        )
        logger.warning(
            "Decreasing %s from %s to %s to fit vLLM max_model_len=%s",
            max_tokens_key,
            requested_max_tokens,
            remaining,
            max_model_len,
        )
        return (
            json.dumps(adjusted_body).encode("utf-8"),
            adjusted_body,
            {_CONTEXT_BUDGET_METADATA_KEY: context_metadata},
            None,
            None,
        )

    @staticmethod
    def _build_context_error_body(
        *,
        request_body: dict[str, Any],
        tokenized: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_tokens = int(metadata["prompt_tokens"])
        max_model_len = int(metadata["max_model_len"])
        message = (
            f"Prompt length {prompt_tokens} leaves no generation budget within "
            f"model context length {max_model_len}."
        )
        return {
            "error": {
                "message": message,
                "type": "context_length_exceeded",
                "code": _MAX_PROMPT_LENGTH_EXCEEDED,
            },
            "model": request_body.get("model", ""),
            "prompt_token_ids": tokenized.get("tokens", []),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": _MAX_PROMPT_LENGTH_EXCEEDED,
                    "stop_reason": _MAX_PROMPT_LENGTH_EXCEEDED,
                    "token_ids": [],
                    "logprobs": {"content": []},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
        }

    async def handle(self, request: Request) -> Response:
        """Proxy *request* to an inference worker, capture trace, return response."""
        await self._ensure_started()
        session_id: str | None = request.state.session_id
        originally_requested_logprobs: bool = getattr(request.state, "originally_requested_logprobs", False)
        body = await request.body()

        try:
            request_body = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_body = {}

        is_stream = request_body.get("stream", False)

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
        t0 = time.perf_counter()
        trace_metadata: dict[str, Any] = {}

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
                (
                    forward_body,
                    request_body,
                    trace_metadata,
                    early_response_body,
                    early_status_code,
                ) = await self._prepare_chat_context_budget(
                    worker_url=worker.url,
                    path=request.url.path,
                    request_body=request_body,
                    raw_body=raw_body,
                    headers=headers,
                )
                if early_response_body is not None:
                    response_body = early_response_body
                    status_code = early_status_code or 400
                else:
                    resp = await self._send_with_retry(
                        method=request.method,
                        url=url,
                        content=forward_body,
                        headers=headers,
                    )
                    content = resp.content
                    status_code = resp.status_code

                    # Parse response for trace extraction
                    try:
                        response_body = json.loads(content)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        response_body = {}
            finally:
                self.router.release(worker.url)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist trace
        if session_id and response_body:
            trace = build_trace_record(
                session_id,
                request_body,
                response_body,
                latency_ms,
                metadata=trace_metadata,
            )
            await self._persist(trace)

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
    # Streaming (SSE)
    # ------------------------------------------------------------------

    async def _handle_streaming(
        self,
        request: Request,
        raw_body: bytes,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool = False,
    ) -> Response:
        if self.local_handler is not None:
            return await self._handle_streaming_local(request_body, session_id, originally_requested_logprobs)

        worker = self.router.route(session_id)
        url = self._build_url(worker.api_url, request.url.path, str(request.url.query))
        headers = self._forward_headers(request)
        trace_metadata: dict[str, Any] = {}

        try:
            (
                raw_body,
                request_body,
                trace_metadata,
                early_response_body,
                early_status_code,
            ) = await self._prepare_chat_context_budget(
                worker_url=worker.url,
                path=request.url.path,
                request_body=request_body,
                raw_body=raw_body,
                headers=headers,
            )
        except Exception:
            self.router.release(worker.url)
            raise

        if early_response_body is not None:
            latency_ms = 0.0
            if session_id:
                trace = build_trace_record(
                    session_id,
                    request_body,
                    early_response_body,
                    latency_ms,
                    metadata=trace_metadata,
                )
                await self._persist(trace)
            self.router.release(worker.url)
            sanitized = strip_vllm_fields(early_response_body) if self.strip_vllm else early_response_body
            if not originally_requested_logprobs:
                sanitized = _strip_logprobs(sanitized)
            return Response(
                content=json.dumps(sanitized),
                status_code=early_status_code or 400,
                media_type="application/json",
            )

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
                if session_id and chunks:
                    trace = build_trace_record_from_chunks(
                        session_id,
                        request_body,
                        chunks,
                        latency_ms,
                        metadata=trace_metadata,
                    )
                    if self.sync_traces:
                        # MemoryStore writes are ~instant; persist inline so
                        # flush is always a no-op and traces are immediately
                        # available without blocking other rollouts.
                        await self.store.store_trace(
                            trace.trace_id, trace.session_id, trace.model_dump()
                        )
                    else:
                        # For real-IO stores (sqlite) use fire-and-forget to
                        # avoid blocking during GeneratorExit.
                        task = asyncio.create_task(
                            self._safe_store(
                                trace.trace_id,
                                trace.session_id,
                                trace.model_dump(),
                            )
                        )
                        self._pending_traces.add(task)
                        task.add_done_callback(self._pending_traces.discard)

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
    ) -> StreamingResponse:
        """Handle streaming when using a local handler (fake-streaming)."""
        assert self.local_handler is not None
        t0 = time.perf_counter()
        response_body = await self.local_handler(request_body)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist trace from the full response
        if session_id and response_body:
            trace = build_trace_record(session_id, request_body, response_body, latency_ms)
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
