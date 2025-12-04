"""Unified OpenAI chat clients with session tracking and optional proxy support.

This module provides TrackedChatClient which inherits from OpenAI and overrides
the `request()` method to intercept ALL API calls for tracing.

Architecture:
    TrackedChatClient(OpenAI) - inherits from OpenAI, overrides request()
    All API calls (chat.completions.create, beta.chat.completions.parse,
    with_raw_response variants, streaming, etc.) go through request().

Aliases (backward compatible):
- ProxyTrackedChatClient = TrackedChatClient (defaults)
- OpenTelemetryTrackedChatClient = TrackedChatClient(enable_local_tracing=False)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai._models import FinalRequestOptions

from rllm.sdk.chat.util import extract_completion_tokens, extract_usage_tokens
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url
from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name
from rllm.sdk.session.contextvar import get_active_cv_sessions
from rllm.sdk.tracers import InMemorySessionTracer

# Shared tracer instance for all clients (when no custom tracer provided)
_SHARED_TRACER = InMemorySessionTracer()


# =============================================================================
# Core tracing logic
# =============================================================================


def _log_trace(
    tracer: Any,
    *,
    model: str | None,
    messages: list[dict] | None,
    response: dict,
    metadata: dict,
    latency_ms: float,
) -> None:
    """Log LLM call to tracer."""
    if not tracer:
        return

    ctx_metadata = {**get_current_metadata(), **metadata}
    token_ids = extract_completion_tokens(response)
    if token_ids:
        ctx_metadata["token_ids"] = {"prompt": [], "completion": token_ids}

    tracer.log_llm_call(
        name="chat.completions.create",
        model=model or "unknown",
        input={"messages": messages or []},
        output=response,
        metadata=ctx_metadata,
        latency_ms=latency_ms,
        tokens=extract_usage_tokens(response),
        trace_id=response.get("id"),
        session_name=get_current_session_name(),
        session_uids=get_active_session_uids(),
        sessions=get_active_cv_sessions(),
    )


def _extract_response_dict(response: Any) -> dict:
    """Extract response dict from various response types."""
    if response is None:
        return {}

    # Direct response objects (ChatCompletion, ParsedChatCompletion, etc.)
    if hasattr(response, "model_dump"):
        return response.model_dump()

    # LegacyAPIResponse from with_raw_response
    if hasattr(response, "parse"):
        try:
            parsed = response.parse()
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump()
        except Exception:
            pass

    return {}


def _is_streaming_response(response: Any) -> bool:
    """Check if response is a streaming response."""
    type_name = type(response).__name__
    return "Stream" in type_name


def _is_chat_completions_endpoint(url: str) -> bool:
    """Check if URL is a chat completions endpoint we should trace."""
    return "/chat/completions" in url


def _accumulate_chunks(chunks: list[Any], model: str | None) -> dict:
    """Reconstruct a response dict from accumulated stream chunks."""
    if not chunks:
        return {}

    first = chunks[0]
    result: dict[str, Any] = {
        "id": getattr(first, "id", None),
        "model": getattr(first, "model", model),
        "object": "chat.completion",
        "created": getattr(first, "created", None),
    }

    # Accumulate content from all chunks
    content_parts: list[str] = []
    role = "assistant"
    finish_reason = None
    tool_call_map: dict[int, dict] = {}

    for chunk in chunks:
        if not getattr(chunk, "choices", None):
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        if delta:
            if getattr(delta, "role", None):
                role = delta.role
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
            # Handle tool calls streaming
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_map:
                        tool_call_map[idx] = {
                            "id": tc.id or "",
                            "type": tc.type or "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        tool_call_map[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_map[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_call_map[idx]["function"]["arguments"] += tc.function.arguments
        if getattr(chunk.choices[0], "finish_reason", None):
            finish_reason = chunk.choices[0].finish_reason

    # Build tool_calls list from map
    tool_calls = []
    if tool_call_map:
        tool_calls = [tool_call_map[i] for i in sorted(tool_call_map.keys())]

    # Build message
    message: dict[str, Any] = {
        "role": role,
        "content": "".join(content_parts) if content_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    result["choices"] = [{"index": 0, "message": message, "finish_reason": finish_reason}]

    # Check for usage in last chunk
    last = chunks[-1]
    if hasattr(last, "usage") and last.usage:
        result["usage"] = {
            "prompt_tokens": last.usage.prompt_tokens or 0,
            "completion_tokens": last.usage.completion_tokens or 0,
            "total_tokens": last.usage.total_tokens or 0,
        }

    return result


# =============================================================================
# Stream wrapper for tracing
# =============================================================================


class TrackedStream:
    """Wrapper around OpenAI Stream that logs trace when fully consumed."""

    def __init__(
        self,
        stream: Any,
        tracer: Any,
        model: str | None,
        messages: list[dict] | None,
        metadata: dict,
        start_time: float,
        enable_tracing: bool,
    ):
        self._stream = stream
        self._tracer = tracer
        self._model = model
        self._messages = messages
        self._metadata = metadata
        self._start_time = start_time
        self._enable_tracing = enable_tracing
        self._chunks: list[Any] = []
        self._exhausted: bool = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            self._chunks.append(chunk)
            return chunk
        except StopIteration:
            self._on_stream_end()
            raise

    def __enter__(self) -> TrackedStream:
        return self

    def __exit__(self, *args: Any) -> None:
        self._stream.__exit__(*args)

    def _on_stream_end(self) -> None:
        """Called when stream is exhausted - logs the trace."""
        if self._exhausted or not self._enable_tracing:
            return
        self._exhausted = True

        latency_ms = (time.perf_counter() - self._start_time) * 1000
        resp_dict = _accumulate_chunks(self._chunks, self._model)

        _log_trace(
            self._tracer,
            model=self._model,
            messages=self._messages,
            response=resp_dict,
            metadata=self._metadata,
            latency_ms=latency_ms,
        )

    # Proxy all other attributes to the underlying stream
    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncTrackedStream:
    """Async wrapper around OpenAI AsyncStream that logs trace when fully consumed."""

    def __init__(
        self,
        stream: Any,
        tracer: Any,
        model: str | None,
        messages: list[dict] | None,
        metadata: dict,
        start_time: float,
        enable_tracing: bool,
    ):
        self._stream = stream
        self._tracer = tracer
        self._model = model
        self._messages = messages
        self._metadata = metadata
        self._start_time = start_time
        self._enable_tracing = enable_tracing
        self._chunks: list[Any] = []
        self._exhausted: bool = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            await self._on_stream_end()
            raise

    async def __aenter__(self) -> AsyncTrackedStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._stream.__aexit__(*args)

    async def _on_stream_end(self) -> None:
        """Called when stream is exhausted - logs the trace."""
        if self._exhausted or not self._enable_tracing:
            return
        self._exhausted = True

        latency_ms = (time.perf_counter() - self._start_time) * 1000
        resp_dict = _accumulate_chunks(self._chunks, self._model)

        _log_trace(
            self._tracer,
            model=self._model,
            messages=self._messages,
            response=resp_dict,
            metadata=self._metadata,
            latency_ms=latency_ms,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# =============================================================================
# Main TrackedChatClient (inherits from OpenAI)
# =============================================================================


class TrackedChatClient(OpenAI):
    """OpenAI client with automatic tracing via request() override.

    Inherits from OpenAI and overrides the request() method to intercept
    ALL API calls for tracing. This is cleaner than patching.

    Args:
        use_proxy: Whether to inject routing metadata in URL (default: True)
        enable_local_tracing: Whether to log traces locally (default: True)
        tracer: Custom tracer instance (default: shared InMemorySessionTracer)
        trace_metadata: Additional metadata to include in traces
        **kwargs: Passed to OpenAI.__init__() (api_key, base_url, etc.)

    Usage:
        # Simple usage
        client = TrackedChatClient()
        response = client.chat.completions.create(...)

        # With proxy
        client = TrackedChatClient(base_url="http://localhost:8000")

        # All OpenAI APIs work automatically
        client.beta.chat.completions.parse(...)
        client.with_raw_response.chat.completions.create(...)
    """

    # Custom attributes we add (not part of OpenAI)
    _use_proxy: bool
    _enable_local_tracing: bool
    _tracer: Any
    _trace_metadata: dict
    _proxy_base_url: str | None

    def __init__(
        self,
        *,
        use_proxy: bool = True,
        enable_local_tracing: bool = True,
        tracer: Any = None,
        trace_metadata: dict | None = None,
        **kwargs: Any,
    ):
        # Store our custom attributes before calling super().__init__
        # We need to be careful here because OpenAI's __init__ may access attributes
        self._use_proxy = use_proxy
        self._enable_local_tracing = enable_local_tracing
        self._tracer = tracer or _SHARED_TRACER
        self._trace_metadata = trace_metadata or {}
        # Store base_url for proxy routing
        self._proxy_base_url = kwargs.get("base_url")

        # Initialize the OpenAI client
        super().__init__(**kwargs)

    def request(
        self,
        cast_to: Any,
        options: FinalRequestOptions,
        *,
        stream: bool = False,
        stream_cls: Any = None,
    ) -> Any:
        """Override request() to add tracing and proxy routing to ALL API calls."""
        # Extract request info for tracing
        model = None
        messages = None
        if options.json_data and isinstance(options.json_data, dict):
            model = options.json_data.get("model")
            messages = options.json_data.get("messages")

        start_time = time.perf_counter()

        # Inject proxy routing metadata if enabled
        if self._use_proxy and self._proxy_base_url:
            routing_metadata = assemble_routing_metadata()
            if routing_metadata:
                new_base_url = build_proxied_base_url(self._proxy_base_url, routing_metadata)
                # Create a temporary client with the proxied URL and make request through it
                # We use with_options() which returns a new client with the modified base_url
                temp_client = OpenAI.with_options(self, base_url=new_base_url)
                # Call the parent's request on the temp client (bypasses our override)
                response = OpenAI.request(temp_client, cast_to, options, stream=stream, stream_cls=stream_cls)
            else:
                response = super().request(cast_to, options, stream=stream, stream_cls=stream_cls)
        else:
            response = super().request(cast_to, options, stream=stream, stream_cls=stream_cls)

        # Log trace if enabled and this is a chat completions endpoint
        if _is_chat_completions_endpoint(options.url) and self._enable_local_tracing:
            if _is_streaming_response(response):
                return TrackedStream(
                    stream=response,
                    tracer=self._tracer,
                    model=model,
                    messages=messages,
                    metadata=self._trace_metadata,
                    start_time=start_time,
                    enable_tracing=True,
                )
            else:
                latency_ms = (time.perf_counter() - start_time) * 1000
                resp_dict = _extract_response_dict(response)

                _log_trace(
                    self._tracer,
                    model=model,
                    messages=messages,
                    response=resp_dict,
                    metadata=self._trace_metadata,
                    latency_ms=latency_ms,
                )

        return response

    @property
    def with_raw_response(self) -> _WithRawResponseWrapper:
        """Return a wrapper for with_raw_response that adds LangChain compatibility.

        LangChain stores `client` (our TrackedChatClient) directly and calls
        `client.with_raw_response.create()`. OpenAI's OpenAIWithRawResponse
        doesn't have a direct create() method, so we wrap it.
        """
        return _WithRawResponseWrapper(self)


class _WithRawResponseWrapper:
    """Wrapper for with_raw_response that adds LangChain compatibility.

    Provides direct create() method that LangChain expects when it stores
    the full client and calls client.with_raw_response.create().
    """

    def __init__(self, client: TrackedChatClient):
        self._client = client
        # Access the cached_property directly from the parent class
        # We need to call the cached_property's __get__ method
        cached_prop = OpenAI.__dict__["with_raw_response"]
        self._raw_response = cached_prop.__get__(client, type(client))

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Direct create method for LangChain compatibility.

        Routes to chat.completions.create().
        """
        return self._raw_response.chat.completions.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying with_raw_response."""
        return getattr(self._raw_response, name)


class AsyncTrackedChatClient(AsyncOpenAI):
    """Async OpenAI client with automatic tracing via request() override.

    Same as TrackedChatClient but inherits from AsyncOpenAI.
    """

    _use_proxy: bool
    _enable_local_tracing: bool
    _tracer: Any
    _trace_metadata: dict
    _proxy_base_url: str | None

    def __init__(
        self,
        *,
        use_proxy: bool = True,
        enable_local_tracing: bool = True,
        tracer: Any = None,
        trace_metadata: dict | None = None,
        **kwargs: Any,
    ):
        self._use_proxy = use_proxy
        self._enable_local_tracing = enable_local_tracing
        self._tracer = tracer or _SHARED_TRACER
        self._trace_metadata = trace_metadata or {}
        self._proxy_base_url = kwargs.get("base_url")

        super().__init__(**kwargs)

    async def request(
        self,
        cast_to: Any,
        options: FinalRequestOptions,
        *,
        stream: bool = False,
        stream_cls: Any = None,
    ) -> Any:
        """Override request() to add tracing and proxy routing to ALL API calls."""
        model = None
        messages = None
        if options.json_data and isinstance(options.json_data, dict):
            model = options.json_data.get("model")
            messages = options.json_data.get("messages")

        start_time = time.perf_counter()

        # Inject proxy routing metadata if enabled
        if self._use_proxy and self._proxy_base_url:
            routing_metadata = assemble_routing_metadata()
            if routing_metadata:
                new_base_url = build_proxied_base_url(self._proxy_base_url, routing_metadata)
                temp_client = AsyncOpenAI.with_options(self, base_url=new_base_url)
                response = await AsyncOpenAI.request(temp_client, cast_to, options, stream=stream, stream_cls=stream_cls)
            else:
                response = await super().request(cast_to, options, stream=stream, stream_cls=stream_cls)
        else:
            response = await super().request(cast_to, options, stream=stream, stream_cls=stream_cls)

        if _is_chat_completions_endpoint(options.url) and self._enable_local_tracing:
            if _is_streaming_response(response):
                return AsyncTrackedStream(
                    stream=response,
                    tracer=self._tracer,
                    model=model,
                    messages=messages,
                    metadata=self._trace_metadata,
                    start_time=start_time,
                    enable_tracing=True,
                )
            else:
                latency_ms = (time.perf_counter() - start_time) * 1000
                resp_dict = _extract_response_dict(response)

                _log_trace(
                    self._tracer,
                    model=model,
                    messages=messages,
                    response=resp_dict,
                    metadata=self._trace_metadata,
                    latency_ms=latency_ms,
                )

        return response

    @property
    def with_raw_response(self) -> _AsyncWithRawResponseWrapper:
        """Return a wrapper for with_raw_response that adds LangChain compatibility."""
        return _AsyncWithRawResponseWrapper(self)


class _AsyncWithRawResponseWrapper:
    """Async wrapper for with_raw_response that adds LangChain compatibility."""

    def __init__(self, client: AsyncTrackedChatClient):
        self._client = client
        cached_prop = AsyncOpenAI.__dict__["with_raw_response"]
        self._raw_response = cached_prop.__get__(client, type(client))

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        """Direct create method for LangChain compatibility."""
        return await self._raw_response.chat.completions.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying with_raw_response."""
        return getattr(self._raw_response, name)


# =============================================================================
# Backward-compatible aliases
# =============================================================================


def ProxyTrackedChatClient(
    *,
    base_url: str | None = None,
    use_proxy: bool = True,
    enable_local_tracing: bool = True,
    tracer: Any = None,
    metadata: dict | None = None,
    **client_kwargs: Any,
) -> TrackedChatClient:
    """Create a TrackedChatClient with proxy support (default behavior)."""
    return TrackedChatClient(
        base_url=base_url,
        use_proxy=use_proxy,
        enable_local_tracing=enable_local_tracing,
        tracer=tracer,
        trace_metadata=metadata,
        **client_kwargs,
    )


def OpenTelemetryTrackedChatClient(
    *,
    base_url: str | None = None,
    use_proxy: bool = True,
    tracer: Any = None,
    metadata: dict | None = None,
    **client_kwargs: Any,
) -> TrackedChatClient:
    """Create a TrackedChatClient without local tracing (for OTel backend only)."""
    return TrackedChatClient(
        base_url=base_url,
        use_proxy=use_proxy,
        enable_local_tracing=False,
        tracer=tracer,
        trace_metadata=metadata,
        **client_kwargs,
    )


def AsyncProxyTrackedChatClient(
    *,
    base_url: str | None = None,
    use_proxy: bool = True,
    enable_local_tracing: bool = True,
    tracer: Any = None,
    metadata: dict | None = None,
    **client_kwargs: Any,
) -> AsyncTrackedChatClient:
    """Create an AsyncTrackedChatClient with proxy support (default behavior)."""
    return AsyncTrackedChatClient(
        base_url=base_url,
        use_proxy=use_proxy,
        enable_local_tracing=enable_local_tracing,
        tracer=tracer,
        trace_metadata=metadata,
        **client_kwargs,
    )


def AsyncOpenTelemetryTrackedChatClient(
    *,
    base_url: str | None = None,
    use_proxy: bool = True,
    tracer: Any = None,
    metadata: dict | None = None,
    **client_kwargs: Any,
) -> AsyncTrackedChatClient:
    """Create an AsyncTrackedChatClient without local tracing (for OTel backend only)."""
    return AsyncTrackedChatClient(
        base_url=base_url,
        use_proxy=use_proxy,
        enable_local_tracing=False,
        tracer=tracer,
        trace_metadata=metadata,
        **client_kwargs,
    )


# Additional aliases for backward compatibility
TrackedAsyncChatClient = AsyncTrackedChatClient
ProxyTrackedAsyncChatClient = AsyncProxyTrackedChatClient
OpenTelemetryTrackedAsyncChatClient = AsyncOpenTelemetryTrackedChatClient

# Legacy aliases (deprecated)
OpenAIOTelClient = OpenTelemetryTrackedChatClient
AsyncOpenAIOTelClient = AsyncOpenTelemetryTrackedChatClient
