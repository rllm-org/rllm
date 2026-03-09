"""Token ID / logprob extraction from OpenAI-style responses.

Extracted from ``rllm/sdk/data_process.py``.  No dependency on rLLM's
``ModelOutput``, ``Step``, or ``Trajectory`` types — only operates on plain
dicts and produces ``TraceRecord`` instances.
"""

import logging
import time
import uuid
from typing import Any

from rllm_model_gateway.models import TraceRecord

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------------


def extract_prompt_token_ids(response: dict[str, Any]) -> list[int]:
    """Extract ``prompt_token_ids`` from the root of a vLLM response."""
    ids = response.get("prompt_token_ids")
    if ids is None:
        return []
    return list(ids)


def extract_completion_token_ids(response: dict[str, Any]) -> list[int]:
    """Extract completion token IDs from ``choices[0].token_ids`` (vLLM 0.11+)."""
    choices = response.get("choices")
    if not choices:
        return []
    ids = choices[0].get("token_ids")
    if ids is None:
        return []
    return list(ids)


def extract_logprobs(response: dict[str, Any]) -> list[float]:
    """Extract per-token logprobs from ``choices[0].logprobs.content``."""
    choices = response.get("choices")
    if not choices:
        return []

    lp_obj = choices[0].get("logprobs")
    if lp_obj is None:
        return []
    content = lp_obj.get("content")
    if content is None:
        return []
    return [float(entry["logprob"]) for entry in content if entry and entry.get("logprob") is not None]


# ------------------------------------------------------------------
# Streaming accumulation helpers
# ------------------------------------------------------------------


def extract_prompt_token_ids_from_chunk(chunk: dict[str, Any]) -> list[int]:
    """Extract ``prompt_token_ids`` from the *first* SSE chunk (vLLM only)."""
    return extract_prompt_token_ids(chunk)


def extract_delta_token_ids(chunk: dict[str, Any]) -> list[int]:
    """Extract delta ``token_ids`` from a single SSE chunk (vLLM 0.11+)."""
    choices = chunk.get("choices")
    if not choices:
        return []
    ids = choices[0].get("token_ids")
    if ids is None:
        return []
    return list(ids)


def extract_delta_logprobs(chunk: dict[str, Any]) -> list[float]:
    """Extract logprobs from a single SSE chunk's ``choices[0].logprobs.content``."""
    choices = chunk.get("choices")
    if not choices:
        return []
    lp = choices[0].get("logprobs")
    if not lp:
        return []
    content = lp.get("content")
    if not content:
        return []
    return [float(e["logprob"]) for e in content if e and e.get("logprob") is not None]


# ------------------------------------------------------------------
# Response sanitisation
# ------------------------------------------------------------------

_VLLM_ROOT_FIELDS = frozenset(
    {
        "prompt_token_ids",
        "prompt_logprobs",
        "kv_transfer_params",
    }
)

_VLLM_CHOICE_FIELDS = frozenset(
    {
        "token_ids",
        "stop_reason",
    }
)


def strip_vllm_fields(response: dict[str, Any]) -> dict[str, Any]:
    """Remove vLLM-specific fields from a response before returning to the client.

    Returns a new dict without modifying the original — important because the
    gateway captures the full response (with token IDs) for the trace and must
    not have those fields stripped from its copy.
    """
    sanitized = {k: v for k, v in response.items() if k not in _VLLM_ROOT_FIELDS}
    if "choices" in sanitized:
        sanitized["choices"] = [{k: v for k, v in choice.items() if k not in _VLLM_CHOICE_FIELDS} for choice in sanitized["choices"]]
    return sanitized


# ------------------------------------------------------------------
# TraceRecord builder
# ------------------------------------------------------------------


def build_trace_record(
    session_id: str,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    latency_ms: float,
    *,
    metadata: dict[str, Any] | None = None,
) -> TraceRecord:
    """Assemble a ``TraceRecord`` from raw request/response dicts."""
    choices = response_body.get("choices") or []
    first_choice = choices[0] if choices else {}

    usage = response_body.get("usage") or {}
    token_counts = {}
    if "prompt_tokens" in usage:
        token_counts["prompt"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        token_counts["completion"] = usage["completion_tokens"]

    return TraceRecord(
        trace_id=str(uuid.uuid4()),
        session_id=session_id,
        model=request_body.get("model", response_body.get("model", "")),
        messages=request_body.get("messages", []),
        prompt_token_ids=extract_prompt_token_ids(response_body),
        response_message=first_choice.get("message") or first_choice.get("delta") or {},
        completion_token_ids=extract_completion_token_ids(response_body),
        logprobs=extract_logprobs(response_body) or None,
        finish_reason=first_choice.get("finish_reason"),
        latency_ms=latency_ms,
        token_counts=token_counts,
        timestamp=time.time(),
        metadata=metadata or {},
        raw_request=request_body,
        raw_response=response_body,
    )


def build_trace_record_from_chunks(
    session_id: str,
    request_body: dict[str, Any],
    chunks: list[dict[str, Any]],
    latency_ms: float,
    *,
    metadata: dict[str, Any] | None = None,
) -> TraceRecord:
    """Assemble a ``TraceRecord`` from accumulated streaming SSE chunks.

    - ``prompt_token_ids`` comes from the *first* chunk.
    - ``completion_token_ids`` are accumulated deltas across all chunks.
    - ``logprobs`` are accumulated across all chunks.
    - The response message is assembled from ``delta`` fields.
    """
    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    role = ""
    content_parts: list[str] = []
    tool_calls_parts: list[dict[str, Any]] = []
    finish_reason: str | None = None
    model = request_body.get("model", "")
    usage: dict[str, Any] = {}

    for i, chunk in enumerate(chunks):
        if i == 0:
            prompt_ids = extract_prompt_token_ids_from_chunk(chunk)
            model = chunk.get("model", model)

        delta_ids = extract_delta_token_ids(chunk)
        completion_ids.extend(delta_ids)

        delta_lp = extract_delta_logprobs(chunk)
        logprobs.extend(delta_lp)

        choices = chunk.get("choices", [])
        if choices:
            c = choices[0]
            delta = c.get("delta", {})
            if delta.get("role"):
                role = delta["role"]
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("tool_calls"):
                tool_calls_parts.extend(delta["tool_calls"])
            if c.get("finish_reason"):
                finish_reason = c["finish_reason"]

        if chunk.get("usage"):
            usage = chunk["usage"]

    response_message: dict[str, Any] = {"role": role or "assistant"}
    content = "".join(content_parts)
    if content:
        response_message["content"] = content
    if tool_calls_parts:
        response_message["tool_calls"] = tool_calls_parts

    token_counts: dict[str, int] = {}
    if "prompt_tokens" in usage:
        token_counts["prompt"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        token_counts["completion"] = usage["completion_tokens"]

    return TraceRecord(
        trace_id=str(uuid.uuid4()),
        session_id=session_id,
        model=model,
        messages=request_body.get("messages", []),
        prompt_token_ids=prompt_ids,
        response_message=response_message,
        completion_token_ids=completion_ids,
        logprobs=logprobs or None,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        token_counts=token_counts,
        timestamp=time.time(),
        metadata=metadata or {},
        raw_request=request_body,
        raw_response=None,  # Too large for streaming; individual chunks not stored
    )
