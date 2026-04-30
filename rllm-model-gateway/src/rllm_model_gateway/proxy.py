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
from rllm_model_gateway.models import GatewayConfig, TraceRecord
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


class _RllmParserTransport:
    """Convert chat-completions requests to raw-text completions requests.

    Rendering and completion parsing stay delegated to ``rllm.parser``; this
    class only adapts the HTTP transport shape.
    """

    def __init__(self, config: GatewayConfig, parser: Any | None = None) -> None:
        self.config = config
        self.accumulate_reasoning = config.accumulate_reasoning

        if parser is not None:
            self.parser = parser
            return

        if not config.tokenizer_name:
            raise ValueError("tokenizer_name is required for rLLM parser transport")

        from transformers import AutoTokenizer

        from rllm.parser import ChatTemplateParser

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
        self.parser = ChatTemplateParser.get_parser(
            tokenizer,
            disable_thinking=config.disable_thinking,
            multi_turn_extension=config.multi_turn_extension,
        )

        if type(self.parser).parse_completion_text is ChatTemplateParser.parse_completion_text:
            raise ValueError(f"Parser {type(self.parser).__name__} does not implement parse_completion_text")

    def chat_to_completion(self, body: dict[str, Any], *, originally_requested_logprobs: bool) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("parser transport requires a chat messages list")
        if body.get("stream"):
            raise ValueError("parser transport does not support stream=true in v1")
        if body.get("n", 1) != 1:
            raise ValueError("parser transport supports only n=1 in v1")
        if body.get("top_logprobs") is not None and not originally_requested_logprobs:
            raise ValueError("top_logprobs requires logprobs=true")

        self._reject_multimodal(messages)

        tools = body.get("tools") or []
        tool_choice = body.get("tool_choice")
        if tool_choice in (None, "auto"):
            tools_for_prompt = tools
        elif tool_choice == "none":
            tools_for_prompt = []
        else:
            raise ValueError("parser transport supports only omitted, auto, or none tool_choice in v1")

        prompt_text = self.parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools_for_prompt,
            accumulate_reasoning=self.accumulate_reasoning,
        )

        stop_token_ids = sorted(set(body.get("stop_token_ids") or []) | set(getattr(self.parser, "stop_sequences", []) or []))
        completion_logprobs = body["top_logprobs"] if body.get("top_logprobs") is not None else 1

        converted: dict[str, Any] = {
            "model": body.get("model"),
            "prompt": prompt_text,
            "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "stop": body.get("stop"),
            "stop_token_ids": stop_token_ids,
            "logprobs": completion_logprobs,
            "return_token_ids": True,
            "add_special_tokens": False,
        }

        for key in (
            "min_tokens",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "seed",
            "ignore_eos",
            "include_stop_str_in_output",
            "skip_special_tokens",
        ):
            if key in body:
                converted[key] = body[key]

        return {k: v for k, v in converted.items() if v is not None}

    def completion_to_chat(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        if len(choices) != 1:
            raise ValueError("parser transport expected exactly one completion choice")

        choice = choices[0]
        completion_text = choice.get("text")
        if completion_text is None:
            raise ValueError("vLLM completion response missing choices[0].text")

        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise ValueError("vLLM completion response missing choices[0].token_ids")
        token_ids = list(token_ids)

        parsed = self.parser.parse_completion_text(completion_text)
        logprobs = self._normalize_logprobs(choice, token_ids)
        tool_calls = self._tool_calls_to_openai(parsed.get("tool_calls") or [], choice.get("index", 0))

        message: dict[str, Any] = {
            "role": "assistant",
            "content": parsed.get("content", "") or "",
        }
        if parsed.get("reasoning"):
            message["reasoning"] = parsed["reasoning"]
        if tool_calls:
            message["tool_calls"] = tool_calls

        prompt_token_ids = choice.get("prompt_token_ids")
        if prompt_token_ids is None:
            prompt_token_ids = response.get("prompt_token_ids", [])

        return {
            "id": response.get("id"),
            "object": "chat.completion",
            "created": response.get("created"),
            "model": response.get("model"),
            "prompt_token_ids": list(prompt_token_ids or []),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else choice.get("finish_reason"),
                    "token_ids": token_ids,
                    "logprobs": logprobs,
                }
            ],
            "usage": response.get("usage", {}),
        }

    @staticmethod
    def _reject_multimodal(messages: list[dict[str, Any]]) -> None:
        for message in messages:
            if message.get("images") is not None:
                raise ValueError("parser transport does not support image inputs in v1")
            content = message.get("content")
            if isinstance(content, list | dict):
                raise ValueError("parser transport does not support multimodal message content in v1")

    @staticmethod
    def _normalize_logprobs(choice: dict[str, Any], token_ids: list[int]) -> dict[str, Any]:
        lp_obj = choice.get("logprobs")
        if not lp_obj:
            raise ValueError("vLLM completion response missing logprobs")

        tokens = lp_obj.get("tokens")
        token_logprobs = lp_obj.get("token_logprobs")
        top_logprobs = lp_obj.get("top_logprobs")
        if tokens is None or token_logprobs is None:
            raise ValueError("vLLM completion response missing sampled-token logprobs")
        if len(tokens) != len(token_ids) or len(token_logprobs) != len(token_ids):
            raise ValueError("vLLM completion logprobs are not aligned with token_ids")
        if any(logprob is None for logprob in token_logprobs):
            raise ValueError("vLLM completion logprobs contain None")

        content: list[dict[str, Any]] = []
        for i, (token, logprob) in enumerate(zip(tokens, token_logprobs, strict=True)):
            entry: dict[str, Any] = {"token": token, "logprob": logprob}
            if top_logprobs is not None and i < len(top_logprobs) and top_logprobs[i] is not None:
                entry["top_logprobs"] = [{"token": top_token, "logprob": top_logprob} for top_token, top_logprob in top_logprobs[i].items()]
            content.append(entry)
        return {"content": content}

    @staticmethod
    def _tool_calls_to_openai(tool_calls: list[Any], choice_index: int) -> list[dict[str, Any]]:
        result = []
        for tool_index, tool_call in enumerate(tool_calls):
            name = getattr(tool_call, "name", None)
            arguments = getattr(tool_call, "arguments", None)
            if name is None and isinstance(tool_call, dict):
                name = tool_call.get("name")
                arguments = tool_call.get("arguments")
            result.append(
                {
                    "id": f"call_{choice_index}_{tool_index}",
                    "type": "function",
                    "function": {"name": name or "", "arguments": json.dumps(arguments or {})},
                }
            )
        return result


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
        parser_config: GatewayConfig | None = None,
        parser_transport: _RllmParserTransport | None = None,
    ) -> None:
        self.router = router
        self.store = store
        self.strip_vllm = strip_vllm
        self.sync_traces = sync_traces
        self.max_retries = max_retries
        self.local_handler = local_handler
        self.parser_transport = parser_transport or (_RllmParserTransport(parser_config) if parser_config is not None else None)
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
            if self.parser_transport is not None and request.method.upper() == "POST" and request.url.path.endswith("/chat/completions"):
                raise ValueError("parser transport does not support stream=true in v1")
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

        if self.local_handler is not None:
            # In-process path: call handler directly, no HTTP
            response_body = await self.local_handler(request_body)
            status_code = 200
        else:
            # HTTP proxy path
            parser_mode = self.parser_transport is not None and request.method.upper() == "POST" and request.url.path.endswith("/chat/completions")
            if parser_mode:
                upstream_body = self.parser_transport.chat_to_completion(
                    request_body,
                    originally_requested_logprobs=originally_requested_logprobs,
                )
                upstream_path = "/v1/completions"
                upstream_content = json.dumps(upstream_body).encode("utf-8")
            else:
                upstream_path = request.url.path
                upstream_content = raw_body

            worker = self.router.route(session_id)
            url = self._build_url(worker.api_url, upstream_path, str(request.url.query))
            headers = self._forward_headers(request)
            try:
                resp = await self._send_with_retry(
                    method=request.method,
                    url=url,
                    content=upstream_content,
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

            if parser_mode and status_code < 400 and response_body:
                response_body = self.parser_transport.completion_to_chat(response_body)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist trace
        if session_id and response_body:
            trace = build_trace_record(session_id, request_body, response_body, latency_ms)
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
                    trace = build_trace_record_from_chunks(session_id, request_body, chunks, latency_ms)
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
