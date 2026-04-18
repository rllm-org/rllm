"""Normalized request/response types passed to/from adapters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool", "developer"]


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Original arguments string emitted by the model, unparsed. Matches the
    # OpenAI chat completions spec (function.arguments is a string that may
    # be invalid JSON). Anthropic emits a dict directly; we json.dumps it
    # into arguments_raw on conversion.
    arguments_raw: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    # str for plain text; list[dict] for multimodal / content-block content.
    content: str | list[dict[str, Any]] | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class NormalizedRequest(BaseModel):
    """Gateway → adapter input."""

    # Chat mode (chat completions / responses / anthropic flatten here)
    messages: list[Message] | None = None
    tools: list[ToolSpec] | None = None
    # Completion mode (legacy /v1/completions)
    prompt: str | None = None
    # Common
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    # Identifier the adapter can use as a cache key (e.g. for prefix-cache
    # affinity in its own worker pool).
    session_id: str = ""


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


class NormalizedResponse(BaseModel):
    """Adapter → gateway output. Also the response shape stored in TraceRecord."""

    content: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "stop"
    usage: Usage = Field(default_factory=Usage)
    # Adapter-emitted extras (e.g. prompt_ids, completion_ids, logprobs,
    # prompt_logprobs, routing_matrices). Names follow rllm conventions; see
    # ``rllm_model_gateway.extras`` for the pinned keys. Opaque to the gateway;
    # persisted as a separate blob row keyed by trace_id.
    extras: dict[str, Any] = Field(default_factory=dict)

    # Small per-request annotations the adapter wants to attach to the trace.
    # Stored inline on the traces table — keep values small. For large blobs
    # (token IDs, logprobs, MoE matrices) use ``extras``.
    metrics: dict[str, float] = Field(default_factory=dict)  # numeric measurements (queue_time_ms, kv_cache_hit_rate, …)
    metadata: dict[str, Any] = Field(default_factory=dict)  # free-form tags (worker_id, backend, model_revision, …)


class AdapterError(Exception):
    """Raise from an adapter to return a specific HTTP status to the agent.

    Adapters that need to express anything more specific than an opaque 502
    (rate-limited upstream → 429, malformed input the engine rejected → 400,
    quota exceeded → 402, etc.) should raise this. Other exceptions still
    become 502 with the exception's str().
    """

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
