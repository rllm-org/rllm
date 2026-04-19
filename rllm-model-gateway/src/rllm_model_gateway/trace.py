"""TraceRecord — one captured LLM call. Stored per-request in the trace store."""

from __future__ import annotations

import time
import uuid
from typing import Any

import msgpack
from pydantic import BaseModel, Field

from rllm_model_gateway.normalized import (
    Message,
    NormalizedRequest,
    NormalizedResponse,
    ToolCall,
    ToolSpec,
    Usage,
)


class TraceRecord(BaseModel):
    """One captured LLM call (request + response, no extras)."""

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    endpoint: str  # "chat_completions" | "completions" | "responses" | "anthropic_messages"
    model: str = ""

    # Request (from NormalizedRequest)
    messages: list[Message] | None = None
    prompt: str | None = None  # legacy completions
    tools: list[ToolSpec] | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # Response (from NormalizedResponse)
    text: str | None = None
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)

    # Adapter-emitted per-trace annotations. Gateway always populates
    # ``metrics["gateway_latency_ms"]`` (end-to-end as seen by the gateway).
    metrics: dict[str, float] = Field(default_factory=dict)  # numeric measurements
    metadata: dict[str, Any] = Field(default_factory=dict)  # free-form tags

    # Adapter-emitted training-side blob (token IDs, logprobs, MoE matrices,
    # etc.). Stored in a separate table internally; populated only when the
    # caller asks for it (e.g. via ``get_traces(..., extras=True)``). When this is
    # ``None`` the caller fetched the lightweight trace; ``{}`` would mean
    # the caller asked for extras and the trace genuinely has none.
    extras: dict[str, Any] | None = None


def build_trace(
    *,
    session_id: str,
    endpoint: str,
    model: str,
    request: NormalizedRequest,
    response: NormalizedResponse,
    gateway_latency_ms: float,
) -> TraceRecord:
    """Assemble a TraceRecord from a normalized req/resp pair.

    ``gateway_latency_ms`` is merged into ``metrics`` under the
    ``gateway_latency_ms`` key alongside any adapter-emitted metrics.
    """
    metrics = dict(response.metrics) if response.metrics else {}
    metrics["gateway_latency_ms"] = gateway_latency_ms
    return TraceRecord(
        session_id=session_id,
        endpoint=endpoint,
        model=model,
        messages=request.messages,
        prompt=request.prompt,
        tools=request.tools,
        kwargs=request.kwargs,
        content=response.content,
        text=response.text,
        reasoning=response.reasoning,
        tool_calls=response.tool_calls,
        finish_reason=response.finish_reason,
        usage=response.usage,
        metrics=metrics,
        metadata=dict(response.metadata) if response.metadata else {},
    )


def serialize_extras(extras: dict[str, Any]) -> tuple[str, bytes] | None:
    """Serialize NormalizedResponse.extras to (format, bytes) for storage.

    Returns None if extras is empty.
    """
    if not extras:
        return None
    return ("msgpack", msgpack.packb(extras, use_bin_type=True))


def deserialize_extras(format: str, data: bytes) -> dict[str, Any]:
    if format == "msgpack":
        return msgpack.unpackb(data, raw=False)
    if format == "json":
        import json

        return json.loads(data.decode("utf-8"))
    raise ValueError(f"Unknown extras format: {format!r}")
