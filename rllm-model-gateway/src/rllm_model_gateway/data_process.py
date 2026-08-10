"""Token ID / logprob extraction from OpenAI-style responses.

Extracted from ``rllm/sdk/data_process.py``.  No dependency on rLLM's
``ModelOutput``, ``Step``, or ``Trajectory`` types — only operates on plain
dicts and produces ``TraceRecord`` instances.
"""

import json
import logging
import time
import uuid
from typing import Any

from rllm_model_gateway.models import TraceRecord

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Anthropic /v1/messages shapes
# ------------------------------------------------------------------

_ANTHROPIC_STREAM_EVENTS = {"message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}


def _is_anthropic_response(response_body: dict[str, Any]) -> bool:
    """True for Anthropic Messages API responses (no OpenAI ``choices``)."""
    return response_body.get("type") == "message" and isinstance(response_body.get("content"), list)


def _anthropic_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic ``tool_use`` block to OpenAI tool_call format."""
    return {
        "id": block.get("id", ""),
        "type": "function",
        "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input") or {})},
    }


def _anthropic_response_message(response_body: dict[str, Any]) -> dict[str, Any]:
    """Synthesize an OpenAI-style assistant message from an Anthropic response."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in response_body.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif block.get("type") == "thinking" and block.get("thinking"):
            reasoning_parts.append(block["thinking"])
        elif block.get("type") == "tool_use":
            tool_calls.append(_anthropic_tool_call(block))
    message: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        message["content"] = "".join(text_parts)
    if reasoning_parts:
        message["reasoning"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


# ------------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------------


def extract_prompt_token_ids(response: dict[str, Any]) -> list[int]:
    """Extract ``prompt_token_ids`` from a vLLM response.

    Checks root level first (chat/completions format), then falls back to
    choices[0].prompt_token_ids (completions format).
    """
    ids = response.get("prompt_token_ids")
    if ids is None:
        choices = response.get("choices")
        if choices:
            ids = choices[0].get("prompt_token_ids")
    return list(ids) if ids is not None else []


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
    """Extract per-token logprobs from a vLLM response.

    Handles both formats:
    - chat/completions: choices[0].logprobs.content[].logprob
    - completions: choices[0].logprobs.token_logprobs (flat list)
    """
    choices = response.get("choices")
    if not choices:
        return []

    lp_obj = choices[0].get("logprobs")
    if lp_obj is None:
        return []

    # Chat/completions format: logprobs.content[].logprob
    content = lp_obj.get("content")
    if content is not None:
        return [float(entry["logprob"]) for entry in content if entry and entry.get("logprob") is not None]

    # Completions format: logprobs.token_logprobs (flat list of floats)
    token_logprobs = lp_obj.get("token_logprobs")
    if token_logprobs is not None:
        return [float(lp) for lp in token_logprobs if lp is not None]

    return []


def extract_weight_version(response: dict[str, Any]) -> int | None:
    """Per-response weight version stamped by the rollout engine.

    Present on rollout-engine outputs (e.g. the Tinker in-process handler);
    absent on plain vLLM responses, which fall back to the proxy's tracked
    version.
    """
    version = response.get("weight_version")
    return int(version) if version is not None else None


def extract_routing_matrices(response: dict[str, Any]) -> list[str] | None:
    """Per-token routing matrices stamped by the rollout engine (R3 router replay).

    Carried on the choice alongside ``token_ids``; absent on plain vLLM responses.
    """
    choices = response.get("choices")
    if not choices:
        return None
    rm = choices[0].get("routing_matrices")
    return list(rm) if rm else None


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
    """Extract logprobs from a single SSE chunk.

    Handles both formats:
    - chat/completions streaming: choices[0].logprobs.content[].logprob
    - completions streaming: choices[0].logprobs.token_logprobs (flat list)
    """
    choices = chunk.get("choices")
    if not choices:
        return []
    lp = choices[0].get("logprobs")
    if not lp:
        return []
    content = lp.get("content")
    if content:
        return [float(e["logprob"]) for e in content if e and e.get("logprob") is not None]
    token_logprobs = lp.get("token_logprobs")
    if token_logprobs:
        return [float(v) for v in token_logprobs if v is not None]
    return []


# ------------------------------------------------------------------
# Response sanitisation
# ------------------------------------------------------------------

_VLLM_ROOT_FIELDS = frozenset(
    {
        "prompt_token_ids",
        "prompt_logprobs",
        "kv_transfer_params",
        "weight_version",
    }
)

_VLLM_CHOICE_FIELDS = frozenset(
    {
        "token_ids",
        "stop_reason",
        "routing_matrices",
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
    weight_version: int | None = None,
    lineage_id: str | None = None,
    trace_id: str | None = None,
    capture_raw: bool = False,
) -> TraceRecord:
    """Assemble a ``TraceRecord`` from raw request/response dicts.

    ``capture_raw`` retains the full ``raw_request``/``raw_response`` on the
    record. Defaults to False: training reads only the token-id / logprob /
    message fields (see ``rllm.engine.trace_converter``), and keeping the raw
    dicts (a ≤120K-token prompt + its full response) balloons ``model_dump``
    serialization on the gateway's event loop — the dominant per-request CPU
    cost at high concurrency. Enable only for debugging.
    """
    choices = response_body.get("choices") or []
    first_choice = choices[0] if choices else {}

    usage = response_body.get("usage") or {}
    token_counts = {}
    if "prompt_tokens" in usage:
        token_counts["prompt"] = usage["prompt_tokens"]
    elif "input_tokens" in usage:  # Anthropic Messages API
        token_counts["prompt"] = usage["input_tokens"]
    if "completion_tokens" in usage:
        token_counts["completion"] = usage["completion_tokens"]
    elif "output_tokens" in usage:  # Anthropic Messages API
        token_counts["completion"] = usage["output_tokens"]

    response_message = first_choice.get("message") or first_choice.get("delta") or {}
    finish_reason = first_choice.get("finish_reason")
    if _is_anthropic_response(response_body):
        response_message = _anthropic_response_message(response_body)
        finish_reason = finish_reason or response_body.get("stop_reason")

    # Proxy's fanned-out version wins; the engine-stamped response value is only a fallback.
    # (A multi-worker subprocess's rebuilt engine stamps a stale version that must not override it.)
    if weight_version is None:
        weight_version = extract_weight_version(response_body)

    return TraceRecord(
        trace_id=trace_id or str(uuid.uuid4()),
        session_id=session_id,
        lineage_id=lineage_id,
        model=request_body.get("model", response_body.get("model", "")),
        messages=request_body.get("messages", []),
        prompt_token_ids=extract_prompt_token_ids(response_body),
        response_message=response_message,
        completion_token_ids=extract_completion_token_ids(response_body),
        logprobs=extract_logprobs(response_body) or None,
        routing_matrices=extract_routing_matrices(response_body),
        finish_reason=finish_reason,
        weight_version=weight_version,
        latency_ms=latency_ms,
        token_counts=token_counts,
        timestamp=time.time(),
        metadata=metadata or {},
        raw_request=request_body if capture_raw else None,
        raw_response=response_body if capture_raw else None,
    )


def build_trace_record_from_chunks(
    session_id: str,
    request_body: dict[str, Any],
    chunks: list[dict[str, Any]],
    latency_ms: float,
    *,
    metadata: dict[str, Any] | None = None,
    weight_version: int | None = None,
    lineage_id: str | None = None,
    trace_id: str | None = None,
    capture_raw: bool = False,
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

    # Anthropic Messages API streams a different event vocabulary
    # (message_start / content_block_* / message_delta) with no ``choices``.
    if any(chunk.get("type") in _ANTHROPIC_STREAM_EVENTS for chunk in chunks):
        return _build_trace_record_from_anthropic_chunks(
            session_id,
            request_body,
            chunks,
            latency_ms,
            metadata=metadata,
            weight_version=weight_version,
            lineage_id=lineage_id,
            trace_id=trace_id,
            capture_raw=capture_raw,
        )

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
        trace_id=trace_id or str(uuid.uuid4()),
        session_id=session_id,
        lineage_id=lineage_id,
        model=model,
        messages=request_body.get("messages", []),
        prompt_token_ids=prompt_ids,
        response_message=response_message,
        completion_token_ids=completion_ids,
        logprobs=logprobs or None,
        finish_reason=finish_reason,
        weight_version=weight_version,
        latency_ms=latency_ms,
        token_counts=token_counts,
        timestamp=time.time(),
        metadata=metadata or {},
        raw_request=request_body if capture_raw else None,
        raw_response=None,  # Too large for streaming; individual chunks not stored
    )


def _build_trace_record_from_anthropic_chunks(
    session_id: str,
    request_body: dict[str, Any],
    chunks: list[dict[str, Any]],
    latency_ms: float,
    *,
    metadata: dict[str, Any] | None = None,
    weight_version: int | None = None,
    lineage_id: str | None = None,
    trace_id: str | None = None,
    capture_raw: bool = False,
) -> TraceRecord:
    """Assemble a ``TraceRecord`` from Anthropic Messages API SSE events.

    Events: ``message_start`` (message shell + input usage),
    ``content_block_start``/``content_block_delta``/``content_block_stop``
    (per-index text / thinking / tool_use assembly), and ``message_delta``
    (stop_reason + output usage). Token IDs are unavailable on this path —
    Anthropic-format upstreams don't return them.
    """
    model = request_body.get("model", "")
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    blocks: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        event = chunk.get("type")
        if event == "message_start":
            message = chunk.get("message") or {}
            model = message.get("model", model)
            start_usage = message.get("usage") or {}
            if "input_tokens" in start_usage:
                input_tokens = start_usage["input_tokens"]
        elif event == "content_block_start":
            block = dict(chunk.get("content_block") or {})
            block.setdefault("type", "text")
            blocks[chunk.get("index", len(blocks))] = block
        elif event == "content_block_delta":
            block = blocks.setdefault(chunk.get("index", len(blocks)), {"type": "text"})
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta.get("type") == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif delta.get("type") == "input_json_delta":
                block["_partial_json"] = block.get("_partial_json", "") + delta.get("partial_json", "")
        elif event == "content_block_stop":
            block = blocks.get(chunk.get("index", -1))
            if block is not None and "_partial_json" in block:
                raw = block.pop("_partial_json")
                try:
                    block["input"] = json.loads(raw or "{}")
                except ValueError:
                    block["input"] = {"_raw": raw}
        elif event == "message_delta":
            delta = chunk.get("delta") or {}
            if delta.get("stop_reason"):
                finish_reason = delta["stop_reason"]
            delta_usage = chunk.get("usage") or {}
            if "output_tokens" in delta_usage:
                output_tokens = delta_usage["output_tokens"]

    content = [blocks[i] for i in sorted(blocks)]
    response_message = _anthropic_response_message({"type": "message", "content": content})

    token_counts: dict[str, int] = {}
    if input_tokens is not None:
        token_counts["prompt"] = input_tokens
    if output_tokens is not None:
        token_counts["completion"] = output_tokens

    return TraceRecord(
        trace_id=trace_id or str(uuid.uuid4()),
        session_id=session_id,
        lineage_id=lineage_id,
        model=model,
        messages=request_body.get("messages", []),
        prompt_token_ids=[],
        response_message=response_message,
        completion_token_ids=[],
        logprobs=None,
        finish_reason=finish_reason,
        weight_version=weight_version,
        latency_ms=latency_ms,
        token_counts=token_counts,
        timestamp=time.time(),
        metadata=metadata or {},
        raw_request=request_body if capture_raw else None,
        raw_response=None,
    )
