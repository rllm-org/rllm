"""Unified OpenAI chat clients with session tracking and optional proxy support.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    TrackedChatClient                            │
    │  ┌───────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
    │  │  use_proxy    │  │ enable_local_    │  │    tracer       │  │
    │  │  (default: T) │  │ tracing (def: T) │  │  (optional)     │  │
    │  └───────────────┘  └──────────────────┘  └─────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
          ┌──────────────┐        ┌──────────────┐
          │   Default    │        │  OTel Mode   │
          │  (proxy+log) │        │ (proxy only) │
          └──────────────┘        └──────────────┘
           use_proxy=True          use_proxy=True
           local_trace=True        local_trace=False

Aliases (backward compatible):
- OpenTelemetryTrackedChatClient = TrackedChatClient(enable_local_tracing=False)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from openai import AsyncOpenAI, OpenAI
from openai._streaming import AsyncStream, Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion import Completion

from rllm.sdk.chat.util import extract_completion_tokens, extract_usage_tokens, merge_args
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url
from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name
from rllm.sdk.session.contextvar import get_active_cv_sessions
from rllm.sdk.tracers import InMemorySessionTracer

if TYPE_CHECKING:
    from typing import Literal

_T = TypeVar("_T")

# Shared tracer instance for all clients (when no custom tracer provided)
_SHARED_TRACER = InMemorySessionTracer()


def _get_scoped_client(client, base_url: str | None, metadata: dict | None, headers: dict | None):
    """Apply proxy URL rewriting and extra headers to client."""
    if base_url and metadata:
        client = client.with_options(base_url=build_proxied_base_url(base_url, metadata))
    if headers:
        client = client.with_options(extra_headers=headers)
    return client


def _log_trace(
    tracer,
    *,
    model: str,
    messages: list[dict],
    response: dict,
    token_ids: list[int] | None,
    metadata: dict,
    latency_ms: float,
) -> None:
    """Log LLM call to tracer."""
    if not tracer:
        return

    ctx_metadata = {**get_current_metadata(), **metadata}
    if token_ids:
        ctx_metadata["token_ids"] = {"prompt": [], "completion": token_ids}

    tracer.log_llm_call(
        name="chat.completions.create",
        model=model,
        input={"messages": messages},
        output=response,
        metadata=ctx_metadata,
        latency_ms=latency_ms,
        tokens=extract_usage_tokens(response),
        trace_id=response.get("id"),
        session_name=get_current_session_name(),
        session_uids=get_active_session_uids(),
        sessions=get_active_cv_sessions(),
    )


# =============================================================================
# Streaming Wrappers
# =============================================================================


def _accumulate_stream_content(chunks: list[ChatCompletionChunk]) -> dict:
    """Reconstruct a response dict from accumulated stream chunks for tracing.

    Args:
        chunks: List of ChatCompletionChunk objects accumulated during streaming

    Returns:
        A dict mimicking ChatCompletion structure for logging purposes
    """
    if not chunks:
        return {}

    # Get metadata from first chunk
    first = chunks[0]
    result: dict[str, Any] = {
        "id": first.id,
        "model": first.model,
        "object": "chat.completion",  # Synthesized
        "created": first.created,
    }

    # Accumulate content from all chunks
    content_parts: list[str] = []
    role = "assistant"
    finish_reason = None
    tool_calls: list[dict] = []
    tool_call_map: dict[int, dict] = {}  # Track tool calls by index

    for chunk in chunks:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta:
            if delta.role:
                role = delta.role
            if delta.content:
                content_parts.append(delta.content)
            # Handle tool calls streaming
            if delta.tool_calls:
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
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    # Build tool_calls list from map
    if tool_call_map:
        tool_calls = [tool_call_map[i] for i in sorted(tool_call_map.keys())]

    # Build message
    message: dict[str, Any] = {
        "role": role,
        "content": "".join(content_parts) if content_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    result["choices"] = [
        {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }
    ]

    # Check for usage in last chunk (some providers include it)
    last = chunks[-1]
    if hasattr(last, "usage") and last.usage:
        result["usage"] = {
            "prompt_tokens": last.usage.prompt_tokens or 0,
            "completion_tokens": last.usage.completion_tokens or 0,
            "total_tokens": last.usage.total_tokens or 0,
        }

    return result


def _accumulate_completion_content(chunks: list[Completion]) -> dict:
    """Reconstruct a response dict from accumulated completion stream chunks.

    Args:
        chunks: List of Completion objects accumulated during streaming

    Returns:
        A dict mimicking Completion structure for logging purposes
    """
    if not chunks:
        return {}

    # Get metadata from first chunk
    first = chunks[0]
    result: dict[str, Any] = {
        "id": first.id,
        "model": first.model,
        "object": "text_completion",
        "created": first.created,
    }

    # Accumulate text from all chunks
    text_parts: list[str] = []
    finish_reason = None

    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.text:
            text_parts.append(choice.text)
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    result["choices"] = [
        {
            "index": 0,
            "text": "".join(text_parts),
            "finish_reason": finish_reason,
        }
    ]

    # Check for usage in last chunk
    last = chunks[-1]
    if hasattr(last, "usage") and last.usage:
        result["usage"] = {
            "prompt_tokens": last.usage.prompt_tokens or 0,
            "completion_tokens": last.usage.completion_tokens or 0,
            "total_tokens": last.usage.total_tokens or 0,
        }

    return result


@dataclass
class TrackedStream(Generic[_T]):
    """Wrapper around OpenAI Stream that logs trace when fully consumed.

    This wrapper transparently passes through chunks from the underlying stream
    while accumulating them. When the stream is exhausted, it logs the trace
    with accurate latency and reconstructed response content.
    """

    _stream: Stream[_T]
    _start_time: float
    _tracer: Any
    _model: str
    _messages: list[dict]
    _metadata: dict
    _enable_tracing: bool
    _accumulator: Any  # Callable[[list[_T]], dict]
    _chunks: list[_T] = field(default_factory=list)
    _exhausted: bool = False

    def __iter__(self) -> Iterator[_T]:
        return self

    def __next__(self) -> _T:
        try:
            chunk = next(self._stream)
            self._chunks.append(chunk)
            return chunk
        except StopIteration:
            self._on_stream_end()
            raise

    def _on_stream_end(self) -> None:
        """Called when stream is exhausted - logs the trace."""
        if self._exhausted or not self._enable_tracing:
            return
        self._exhausted = True

        latency_ms = (time.perf_counter() - self._start_time) * 1000
        resp_dict = self._accumulator(self._chunks)

        _log_trace(
            self._tracer,
            model=self._model,
            messages=self._messages,
            response=resp_dict,
            token_ids=extract_completion_tokens(resp_dict),
            metadata=self._metadata,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        """Close the underlying stream."""
        self._on_stream_end()
        self._stream.close()

    def __enter__(self) -> TrackedStream[_T]:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


@dataclass
class TrackedAsyncStream(Generic[_T]):
    """Async wrapper around OpenAI AsyncStream that logs trace when fully consumed."""

    _stream: AsyncStream[_T]
    _start_time: float
    _tracer: Any
    _model: str
    _messages: list[dict]
    _metadata: dict
    _enable_tracing: bool
    _accumulator: Any  # Callable[[list[_T]], dict]
    _chunks: list[_T] = field(default_factory=list)
    _exhausted: bool = False

    def __aiter__(self) -> TrackedAsyncStream[_T]:
        return self

    async def __anext__(self) -> _T:
        try:
            chunk = await self._stream.__anext__()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._on_stream_end()
            raise

    def _on_stream_end(self) -> None:
        """Called when stream is exhausted - logs the trace."""
        if self._exhausted or not self._enable_tracing:
            return
        self._exhausted = True

        latency_ms = (time.perf_counter() - self._start_time) * 1000
        resp_dict = self._accumulator(self._chunks)

        _log_trace(
            self._tracer,
            model=self._model,
            messages=self._messages,
            response=resp_dict,
            token_ids=extract_completion_tokens(resp_dict),
            metadata=self._metadata,
            latency_ms=latency_ms,
        )

    async def close(self) -> None:
        """Close the underlying stream."""
        self._on_stream_end()
        await self._stream.close()

    async def __aenter__(self) -> TrackedAsyncStream[_T]:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# =============================================================================
# Sync Implementation
# =============================================================================


@dataclass
class _ChatCompletionsWithRawResponse:
    """Wrapper for chat.completions.with_raw_response that routes create() and parse() correctly.

    Both create() and parse() make HTTP requests, so both need metadata injection.
    """

    parent: TrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Route to tracked client's with_raw_response.create() for metadata injection."""
        return self.parent.with_raw_response.chat.completions.create(*args, **kwargs)

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        """Route to tracked client's with_raw_response.parse() for metadata injection.

        Note: parse() makes an HTTP request and returns a raw response object that has a .parse() method.
        LangChain calls with_raw_response.parse() to get the raw response, then calls .parse() on it.
        """
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")
        # Extract user-provided metadata (if any) - this is for OpenAI API, not our routing
        metadata = call_kwargs.pop("metadata", None) or {}

        # Get session metadata for proxy routing (this goes in the URL slug)
        routing_metadata = assemble_routing_metadata(metadata if isinstance(metadata, dict) else None)

        # Get client with metadata injection in URL
        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, routing_metadata, p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        # Call with_raw_response.parse() on the metadata-injected client
        start = time.perf_counter()
        raw_response = client.with_raw_response.chat.completions.parse(**call_kwargs)

        # Log trace if enabled
        if p.enable_local_tracing:
            latency_ms = (time.perf_counter() - start) * 1000
            # Handle both RawResponse (has .data) and LegacyAPIResponse (needs .parse())
            if hasattr(raw_response, "data"):
                resp_dict = raw_response.data.model_dump()
            elif hasattr(raw_response, "parse"):
                parsed = raw_response.parse()
                resp_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else {}
            else:
                resp_dict = {}

            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return raw_response


@dataclass
class _ChatCompletions:
    """Namespace for chat.completions.create()"""

    parent: TrackedChatClient

    @property
    def with_raw_response(self) -> _ChatCompletionsWithRawResponse:
        """Return a wrapper that routes create() to tracked client (with metadata)
        and parse() to underlying client (no metadata needed for parsing).
        """
        return _ChatCompletionsWithRawResponse(self.parent)

    @overload
    def create(self, *args: Any, stream: Literal[True], **kwargs: Any) -> TrackedStream[ChatCompletionChunk]: ...

    @overload
    def create(self, *args: Any, stream: Literal[False] = ..., **kwargs: Any) -> ChatCompletion: ...

    @overload
    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | TrackedStream[ChatCompletionChunk]: ...

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | TrackedStream[ChatCompletionChunk]:
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}
        is_streaming = call_kwargs.get("stream", False)

        # Get client (with proxy URL if enabled)
        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, assemble_routing_metadata(metadata), p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        response = client.chat.completions.create(**call_kwargs)

        # Handle streaming responses
        if is_streaming:
            return TrackedStream(
                _stream=response,  # type: ignore[arg-type]
                _start_time=start,
                _tracer=p._tracer or _SHARED_TRACER,
                _model=model or "",
                _messages=list(messages),
                _metadata=metadata,
                _enable_tracing=p.enable_local_tracing,
                _accumulator=_accumulate_stream_content,
            )

        # Non-streaming: log trace immediately
        latency_ms = (time.perf_counter() - start) * 1000
        if p.enable_local_tracing:
            resp_dict = response.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return response


@dataclass
class _Completions:
    """Namespace for completions.create()"""

    parent: TrackedChatClient

    @overload
    def create(self, *args: Any, stream: Literal[True], **kwargs: Any) -> TrackedStream[Completion]: ...

    @overload
    def create(self, *args: Any, stream: Literal[False] = ..., **kwargs: Any) -> Completion: ...

    @overload
    def create(self, *args: Any, **kwargs: Any) -> Completion | TrackedStream[Completion]: ...

    def create(self, *args: Any, **kwargs: Any) -> Completion | TrackedStream[Completion]:
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}
        is_streaming = call_kwargs.get("stream", False)

        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, assemble_routing_metadata(metadata), p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        response = client.completions.create(**call_kwargs)

        # Handle streaming responses
        if is_streaming:
            # Convert prompt to messages format for tracing
            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            return TrackedStream(
                _stream=response,  # type: ignore[arg-type]
                _start_time=start,
                _tracer=p._tracer or _SHARED_TRACER,
                _model=model or "",
                _messages=[{"role": "user", "content": prompt_str}],
                _metadata=metadata,
                _enable_tracing=p.enable_local_tracing,
                _accumulator=_accumulate_completion_content,
            )

        # Non-streaming: log trace immediately
        latency_ms = (time.perf_counter() - start) * 1000
        if p.enable_local_tracing:
            resp_dict = response.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return response


@dataclass
class _ChatNamespace:
    parent: TrackedChatClient

    @property
    def completions(self) -> _ChatCompletions:
        return _ChatCompletions(self.parent)


@dataclass
class _RawChatCompletions:
    """Namespace for with_raw_response.chat.completions.create()"""

    parent: TrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> Any:  # Returns RawResponse
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}

        # Get client (with proxy URL if enabled)
        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, assemble_routing_metadata(metadata), p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        # Call with_raw_response on the underlying client
        raw_response = client.with_raw_response.chat.completions.create(**call_kwargs)

        # Log trace if enabled (using the parsed data from raw_response.data)
        if p.enable_local_tracing:
            latency_ms = (time.perf_counter() - start) * 1000
            resp_dict = raw_response.data.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return raw_response


@dataclass
class _RawChatNamespace:
    parent: TrackedChatClient

    @property
    def completions(self) -> _RawChatCompletions:
        return _RawChatCompletions(self.parent)


@dataclass
class _WithRawResponseWrapper:
    """Wrapper for with_raw_response that preserves proxy/tracing logic"""

    parent: TrackedChatClient

    @property
    def chat(self) -> _RawChatNamespace:
        return _RawChatNamespace(self.parent)

    def create(self, *args: Any, **kwargs: Any) -> Any:  # Returns RawResponse
        """Direct create method for LangChain compatibility.

        LangChain calls client.with_raw_response.create() directly.
        We should call the underlying OpenAI client's with_raw_response directly.
        """
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")
        metadata = call_kwargs.pop("metadata", None) or {}

        # Get client (with proxy URL if enabled)
        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, assemble_routing_metadata(metadata), p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        # Call the underlying OpenAI client's with_raw_response directly
        start = time.perf_counter()
        raw_response = client.with_raw_response.chat.completions.create(**call_kwargs)

        # Handle tracing if enabled (handle both RawResponse and LegacyAPIResponse)
        if p.enable_local_tracing:
            latency_ms = (time.perf_counter() - start) * 1000
            # Handle both RawResponse (has .data) and LegacyAPIResponse (needs .parse())
            if hasattr(raw_response, "data"):
                resp_dict = raw_response.data.model_dump()
            elif hasattr(raw_response, "parse"):
                parsed_response = raw_response.parse()
                resp_dict = parsed_response.model_dump() if hasattr(parsed_response, "model_dump") else parsed_response
            else:
                # Fallback: try to get response directly
                resp_dict = raw_response.model_dump() if hasattr(raw_response, "model_dump") else {}

            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return raw_response


class TrackedChatClient:
    """OpenAI client wrapper with proxy support and tracing.

    Args:
        client: Pre-configured OpenAI client (if None, creates one with **kwargs)
        use_proxy: Inject metadata into proxy URL (default: True)
        enable_local_tracing: Log traces locally (default: True)
        tracer: Custom tracer (default: shared in-memory tracer)
        **kwargs: Passed directly to OpenAI() if client not provided
    """

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        use_proxy: bool = True,
        enable_local_tracing: bool = True,
        tracer: Any = None,
        **kwargs: Any,
    ) -> None:
        # Use provided client or create one (let OpenAI handle all its own args)
        self._client = client if client is not None else OpenAI(**kwargs)

        # Resolve base_url for proxy routing from the client
        # Skip default OpenAI URL (would break if we tried to proxy-rewrite it)
        client_url = getattr(self._client, "base_url", None)
        if client_url and str(client_url).rstrip("/") != "https://api.openai.com/v1":
            self.base_url = str(client_url)
        else:
            self.base_url = None

        self.use_proxy = use_proxy
        self.enable_local_tracing = enable_local_tracing
        self._tracer = tracer
        self._headers: dict[str, str] = {}

        self.chat = _ChatNamespace(self)
        self.completions = _Completions(self)

    @property
    def root_client(self) -> OpenAI:
        """Return the underlying OpenAI client for LangChain compatibility."""
        return self._client

    @property
    def with_raw_response(self) -> _WithRawResponseWrapper:
        """Return a wrapper that provides with_raw_response with tracking/proxy support."""
        return _WithRawResponseWrapper(self)


# =============================================================================
# Async Implementation
# =============================================================================


@dataclass
class _AsyncChatCompletionsWithRawResponse:
    """Wrapper for chat.completions.with_raw_response that routes create() and parse() correctly (async).

    Both create() and parse() make HTTP requests, so both need metadata injection.
    """

    parent: TrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        """Route to tracked client's with_raw_response.create() for metadata injection."""
        return await self.parent.with_raw_response.chat.completions.create(*args, **kwargs)

    async def parse(self, *args: Any, **kwargs: Any) -> Any:
        """Route to tracked client's with_raw_response.parse() for metadata injection.

        Note: parse() makes an HTTP request and returns a raw response object that has a .parse() method.
        LangChain calls with_raw_response.parse() to get the raw response, then calls .parse() on it.
        """
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")
        # Extract user-provided metadata (if any) - this is for OpenAI API, not our routing
        metadata = call_kwargs.pop("metadata", None) or {}

        # Get session metadata for proxy routing (this goes in the URL slug)
        routing_metadata = assemble_routing_metadata(metadata if isinstance(metadata, dict) else None)

        # Get client with metadata injection in URL
        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, routing_metadata, p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        # Call with_raw_response.parse() on the metadata-injected client
        start = time.perf_counter()
        raw_response = await client.with_raw_response.chat.completions.parse(**call_kwargs)

        # Log trace if enabled
        if p.enable_local_tracing:
            latency_ms = (time.perf_counter() - start) * 1000
            # Handle both RawResponse (has .data) and LegacyAPIResponse (needs .parse())
            if hasattr(raw_response, "data"):
                resp_dict = raw_response.data.model_dump()
            elif hasattr(raw_response, "parse"):
                parsed = raw_response.parse()
                resp_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else {}
            else:
                resp_dict = {}

            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return raw_response


@dataclass
class _AsyncChatCompletions:
    parent: TrackedAsyncChatClient

    @property
    def with_raw_response(self) -> _AsyncChatCompletionsWithRawResponse:
        """Return a wrapper that routes create() to tracked client (with metadata)
        and parse() to underlying client (no metadata needed for parsing).
        """
        return _AsyncChatCompletionsWithRawResponse(self.parent)

    @overload
    async def create(self, *args: Any, stream: Literal[True], **kwargs: Any) -> TrackedAsyncStream[ChatCompletionChunk]: ...

    @overload
    async def create(self, *args: Any, stream: Literal[False] = ..., **kwargs: Any) -> ChatCompletion: ...

    @overload
    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | TrackedAsyncStream[ChatCompletionChunk]: ...

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | TrackedAsyncStream[ChatCompletionChunk]:
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}
        is_streaming = call_kwargs.get("stream", False)

        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, assemble_routing_metadata(metadata), p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        response = await client.chat.completions.create(**call_kwargs)

        # Handle streaming responses
        if is_streaming:
            return TrackedAsyncStream(
                _stream=response,  # type: ignore[arg-type]
                _start_time=start,
                _tracer=p._tracer or _SHARED_TRACER,
                _model=model or "",
                _messages=list(messages),
                _metadata=metadata,
                _enable_tracing=p.enable_local_tracing,
                _accumulator=_accumulate_stream_content,
            )

        # Non-streaming: log trace immediately
        latency_ms = (time.perf_counter() - start) * 1000
        if p.enable_local_tracing:
            resp_dict = response.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return response


@dataclass
class _AsyncCompletions:
    parent: TrackedAsyncChatClient

    @overload
    async def create(self, *args: Any, stream: Literal[True], **kwargs: Any) -> TrackedAsyncStream[Completion]: ...

    @overload
    async def create(self, *args: Any, stream: Literal[False] = ..., **kwargs: Any) -> Completion: ...

    @overload
    async def create(self, *args: Any, **kwargs: Any) -> Completion | TrackedAsyncStream[Completion]: ...

    async def create(self, *args: Any, **kwargs: Any) -> Completion | TrackedAsyncStream[Completion]:
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}
        is_streaming = call_kwargs.get("stream", False)

        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, assemble_routing_metadata(metadata), p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        response = await client.completions.create(**call_kwargs)

        # Handle streaming responses
        if is_streaming:
            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            return TrackedAsyncStream(
                _stream=response,  # type: ignore[arg-type]
                _start_time=start,
                _tracer=p._tracer or _SHARED_TRACER,
                _model=model or "",
                _messages=[{"role": "user", "content": prompt_str}],
                _metadata=metadata,
                _enable_tracing=p.enable_local_tracing,
                _accumulator=_accumulate_completion_content,
            )

        # Non-streaming: log trace immediately
        latency_ms = (time.perf_counter() - start) * 1000
        if p.enable_local_tracing:
            resp_dict = response.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return response


@dataclass
class _AsyncChatNamespace:
    parent: TrackedAsyncChatClient

    @property
    def completions(self) -> _AsyncChatCompletions:
        return _AsyncChatCompletions(self.parent)


@dataclass
class _AsyncRawChatCompletions:
    """Namespace for with_raw_response.chat.completions.create() (async)"""

    parent: TrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> Any:  # Returns RawResponse
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}

        if p.use_proxy:
            client = _get_scoped_client(p._client, p.base_url, assemble_routing_metadata(metadata), p._headers)
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        # Call with_raw_response on the underlying client
        raw_response = await client.with_raw_response.chat.completions.create(**call_kwargs)

        # Log trace if enabled (using the parsed data from raw_response.data)
        if p.enable_local_tracing:
            latency_ms = (time.perf_counter() - start) * 1000
            resp_dict = raw_response.data.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return raw_response


@dataclass
class _AsyncRawChatNamespace:
    parent: TrackedAsyncChatClient

    @property
    def completions(self) -> _AsyncRawChatCompletions:
        return _AsyncRawChatCompletions(self.parent)


@dataclass
class _AsyncWithRawResponseWrapper:
    """Wrapper for with_raw_response that preserves proxy/tracing logic (async)"""

    parent: TrackedAsyncChatClient

    @property
    def chat(self) -> _AsyncRawChatNamespace:
        return _AsyncRawChatNamespace(self.parent)

    async def create(self, *args: Any, **kwargs: Any) -> Any:  # Returns RawResponse
        """Direct create method for LangChain compatibility (async).

        LangChain calls client.with_raw_response.create() directly,
        so this method routes to chat.completions.create() internally.
        """
        return await self.chat.completions.create(*args, **kwargs)


class TrackedAsyncChatClient:
    """Async OpenAI client wrapper with proxy support and tracing."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        use_proxy: bool = True,
        enable_local_tracing: bool = True,
        tracer: Any = None,
        **kwargs: Any,
    ) -> None:
        # Use provided client or create one (let AsyncOpenAI handle all its own args)
        self._client = client if client is not None else AsyncOpenAI(**kwargs)

        # Resolve base_url for proxy routing from the client
        # Skip default OpenAI URL (would break if we tried to proxy-rewrite it)
        client_url = getattr(self._client, "base_url", None)
        if client_url and str(client_url).rstrip("/") != "https://api.openai.com/v1":
            self.base_url = str(client_url)
        else:
            self.base_url = None

        self.use_proxy = use_proxy
        self.enable_local_tracing = enable_local_tracing
        self._tracer = tracer
        self._headers: dict[str, str] = {}

        self.chat = _AsyncChatNamespace(self)
        self.completions = _AsyncCompletions(self)

    @property
    def root_client(self) -> AsyncOpenAI:
        """Return the underlying AsyncOpenAI client for LangChain compatibility."""
        return self._client

    @property
    def with_raw_response(self) -> _AsyncWithRawResponseWrapper:
        """Return a wrapper that provides with_raw_response with tracking/proxy support."""
        return _AsyncWithRawResponseWrapper(self)


# =============================================================================
# Backward-compatible Aliases
# =============================================================================
# These are simple subclasses that set sensible defaults for common use cases.
# No logic changes - just preset configurations.


class ProxyTrackedChatClient(TrackedChatClient):
    """Alias: TrackedChatClient with defaults (use_proxy=True, local_tracing=True)"""

    pass


class ProxyTrackedAsyncChatClient(TrackedAsyncChatClient):
    """Alias: TrackedAsyncChatClient with defaults"""

    pass


class OpenTelemetryTrackedChatClient(TrackedChatClient):
    """Alias: TrackedChatClient with enable_local_tracing=False (OTel mode)"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(enable_local_tracing=False, **kwargs)


class OpenTelemetryTrackedAsyncChatClient(TrackedAsyncChatClient):
    """Alias: TrackedAsyncChatClient with enable_local_tracing=False"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(enable_local_tracing=False, **kwargs)


# Legacy shorthand names
OpenAIOTelClient = OpenTelemetryTrackedChatClient
AsyncOpenAIOTelClient = OpenTelemetryTrackedAsyncChatClient
