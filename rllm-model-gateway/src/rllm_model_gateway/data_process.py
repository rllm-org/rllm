"""Token ID / logprob extraction and trace construction for model responses.

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


_ANTHROPIC_STOP_REASONS = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _anthropic_response_fields(response: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Convert an Anthropic message response to the gateway's OpenAI-like trace shape."""
    message: dict[str, Any] = {"role": response.get("role") or "assistant"}
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in response.get("content") or []:
        if block.get("type") == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    if text_parts:
        message["content"] = "".join(text_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_reason = response.get("stop_reason")
    return message, _ANTHROPIC_STOP_REASONS.get(stop_reason, stop_reason)


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
    is_anthropic = response_body.get("type") == "message" and isinstance(response_body.get("content"), list)
    if is_anthropic:
        response_message, finish_reason = _anthropic_response_fields(response_body)
    else:
        response_message = first_choice.get("message") or first_choice.get("delta") or {}
        finish_reason = first_choice.get("finish_reason")

    usage = response_body.get("usage") or {}
    token_counts = {}
    if "prompt_tokens" in usage:
        token_counts["prompt"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        token_counts["completion"] = usage["completion_tokens"]
    if "input_tokens" in usage:
        token_counts["prompt"] = usage["input_tokens"]
    if "output_tokens" in usage:
        token_counts["completion"] = usage["output_tokens"]

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
    """Assemble a ``TraceRecord`` from accumulated OpenAI or Anthropic SSE chunks.

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
    anthropic_tools: dict[int, dict[str, Any]] = {}

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
            usage.update(chunk["usage"])

        event_type = chunk.get("type")
        if event_type == "message_start":
            message = chunk.get("message") or {}
            role = message.get("role", role)
            model = message.get("model", model)
            usage.update(message.get("usage") or {})
        elif event_type == "content_block_start":
            index = chunk.get("index", 0)
            block = chunk.get("content_block") or {}
            if block.get("type") == "text" and block.get("text"):
                content_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                anthropic_tools[index] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input") or {},
                    "partial_json": "",
                }
        elif event_type == "content_block_delta":
            index = chunk.get("index", 0)
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                content_parts.append(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta":
                tool = anthropic_tools.setdefault(index, {"id": "", "name": "", "input": {}, "partial_json": ""})
                tool["partial_json"] += delta.get("partial_json", "")
        elif event_type == "message_delta":
            delta = chunk.get("delta") or {}
            if delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
                finish_reason = _ANTHROPIC_STOP_REASONS.get(stop_reason, stop_reason)

    for tool in anthropic_tools.values():
        arguments = tool["partial_json"] or json.dumps(tool["input"])
        tool_calls_parts.append(
            {
                "id": tool["id"],
                "type": "function",
                "function": {"name": tool["name"], "arguments": arguments},
            }
        )

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
    if "input_tokens" in usage:
        token_counts["prompt"] = usage["input_tokens"]
    if "output_tokens" in usage:
        token_counts["completion"] = usage["output_tokens"]

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
