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
from rllm_model_gateway.metadata import RllmMetadata
from rllm_model_gateway.models import TraceRecord, UpstreamRoute
from rllm_model_gateway.session_router import SessionRouter
from rllm_model_gateway.store.base import TraceStore

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

# Headers stripped before forwarding to an upstream. The client's auth
# headers are replaced with the route's ``auth_header`` (if set), and
# the X-RLLM-* metadata is gateway-internal and never leaks upstream.
_UPSTREAM_FORWARD_STRIP = frozenset(
    {
        "authorization",
        "x-api-key",
        "x-rllm-session-id",
        "x-rllm-run-id",
        "x-rllm-harness",
        "x-rllm-step-id",
        "x-rllm-parent-span-id",
        "x-rllm-project",
        "x-rllm-experiment",
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
        routes: dict[str, UpstreamRoute] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.router = router
        self.store = store
        self.strip_vllm = strip_vllm
        self.sync_traces = sync_traces
        self.max_retries = max_retries
        self.local_handler = local_handler
        self.routes: dict[str, UpstreamRoute] = dict(routes or {})
        # Tag every persisted trace with this run_id so the cross-run
        # viewer can group by gateway lifetime. Empty string is the
        # "unstamped" bucket for callers that don't pass a run_id.
        self._run_id = run_id or ""
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
        rllm_metadata: RllmMetadata | None = getattr(request.state, "rllm_metadata", None)
        route = self._lookup_route(request_body)

        if route is not None:
            url = self._build_url(route.upstream_url, request.url.path, str(request.url.query))
            headers = self._forward_headers(request, extra_strip=_UPSTREAM_FORWARD_STRIP)
            if route.auth_header:
                headers["authorization"] = route.auth_header
            forward_body = self._maybe_drop_params(raw_body, request_body, route.drop_params)
            resp = await self._send_with_retry(
                method=request.method,
                url=url,
                content=forward_body,
                headers=headers,
            )
            content = resp.content
            status_code = resp.status_code
            try:
                response_body = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_body = {}
        elif self.local_handler is not None:
            # In-process path: call handler directly, no HTTP
            response_body = await self.local_handler(request_body)
            status_code = 200
        else:
            # HTTP proxy path (worker pool — training)
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
            trace = build_trace_record(
                session_id,
                request_body,
                response_body,
                latency_ms,
                rllm_metadata=rllm_metadata,
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
    ) -> StreamingResponse:
        rllm_metadata: RllmMetadata | None = getattr(request.state, "rllm_metadata", None)
        route = self._lookup_route(request_body)

        if route is not None:
            return await self._handle_streaming_upstream(
                request,
                raw_body,
                request_body,
                session_id,
                originally_requested_logprobs,
                route,
                rllm_metadata,
            )

        if self.local_handler is not None:
            return await self._handle_streaming_local(request_body, session_id, originally_requested_logprobs)

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
                    trace = build_trace_record_from_chunks(
                        session_id,
                        request_body,
                        chunks,
                        latency_ms,
                        rllm_metadata=rllm_metadata,
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

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            status_code=resp.status_code,
        )

    async def _handle_streaming_upstream(
        self,
        request: Request,
        raw_body: bytes,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool,
        route: UpstreamRoute,
        rllm_metadata: RllmMetadata | None,
    ) -> StreamingResponse:
        """Stream a chat-completions response from a configured upstream.

        Same shape as the worker-pool streaming path but skips ``SessionRouter``
        and uses the upstream's URL + auth header instead.
        """
        url = self._build_url(route.upstream_url, request.url.path, str(request.url.query))
        headers = self._forward_headers(request, extra_strip=_UPSTREAM_FORWARD_STRIP)
        if route.auth_header:
            headers["authorization"] = route.auth_header
        forward_body = self._maybe_drop_params(raw_body, request_body, route.drop_params)

        assert self._http is not None
        upstream = self._http.stream(
            method=request.method,
            url=url,
            content=forward_body,
            headers=headers,
        )
        resp = await upstream.__aenter__()

        t0 = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        needs_strip_vllm = self.strip_vllm
        needs_strip_logprobs = not originally_requested_logprobs

        async def event_generator():
            try:
                async for line in resp.aiter_lines():
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
                    if not line:
                        continue
                    yield line + "\n"
            finally:
                await upstream.__aexit__(None, None, None)
                latency_ms = (time.perf_counter() - t0) * 1000
                if session_id and chunks:
                    trace = build_trace_record_from_chunks(
                        session_id,
                        request_body,
                        chunks,
                        latency_ms,
                        rllm_metadata=rllm_metadata,
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
                await self.store.store_trace(trace.trace_id, trace.session_id, data, self._run_id)
            else:
                task = asyncio.create_task(self._safe_store(trace.trace_id, trace.session_id, data))
                self._pending_traces.add(task)
                task.add_done_callback(self._pending_traces.discard)
        except Exception:
            logger.exception("Failed to persist trace %s", trace.trace_id)

    async def _safe_store(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        try:
            await self.store.store_trace(trace_id, session_id, data, self._run_id)
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
    def _forward_headers(
        request: Request,
        *,
        extra_strip: frozenset[str] = frozenset(),
    ) -> dict[str, str]:
        skip = _HOP_BY_HOP | extra_strip
        return {k: v for k, v in request.headers.items() if k.lower() not in skip}

    def _lookup_route(self, request_body: dict[str, Any]) -> UpstreamRoute | None:
        if not self.routes:
            return None
        model = request_body.get("model") if isinstance(request_body, dict) else None
        if not isinstance(model, str):
            return None
        return self.routes.get(model)

    @staticmethod
    def _maybe_drop_params(
        raw_body: bytes,
        request_body: dict[str, Any],
        drop_params: list[str],
    ) -> bytes:
        """Return ``raw_body`` untouched, or a re-serialised body with *drop_params* removed."""
        if not drop_params or not isinstance(request_body, dict):
            return raw_body
        present = [k for k in drop_params if k in request_body]
        if not present:
            return raw_body
        mutated = {k: v for k, v in request_body.items() if k not in present}
        return json.dumps(mutated).encode("utf-8")
