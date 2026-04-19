"""OpenAI /v1/chat/completions wire format ↔ normalized."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from rllm_model_gateway.normalized import (
    Message,
    NormalizedRequest,
    NormalizedResponse,
    ToolCall,
    ToolSpec,
    Usage,
)

NAME = "chat_completions"
PATH = "/v1/chat/completions"  # FastAPI route inside the gateway
UPSTREAM_PATH = "/chat/completions"  # appended to user-supplied upstream_url

# Wire fields consumed at the gateway level or already captured as structured
# request fields — excluded from the flat kwargs bag.
_RESERVED_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "stream",
        "user",
        "metadata",
        "stream_options",
    }
)


# ---------------------------------------------------------------------------
# Inbound: wire request → NormalizedRequest
# ---------------------------------------------------------------------------


def to_normalized_request(body: dict[str, Any]) -> NormalizedRequest:
    messages = [_msg_from_wire(m) for m in body.get("messages") or []]
    tools = _tools_from_wire(body.get("tools"))
    kwargs = {k: v for k, v in body.items() if k not in _RESERVED_KEYS}
    return NormalizedRequest(
        messages=messages or None,
        tools=tools,
        kwargs=kwargs,
    )


def _msg_from_wire(m: dict[str, Any]) -> Message:
    tcs = m.get("tool_calls")
    parsed_tcs = [_tool_call_from_wire(tc) for tc in tcs] if tcs else None
    return Message(
        role=m["role"],
        content=m.get("content"),
        # vLLM/SGLang emit "reasoning_content"; DeepSeek too. OpenAI uses "reasoning".
        reasoning=m.get("reasoning") or m.get("reasoning_content"),
        tool_calls=parsed_tcs,
        tool_call_id=m.get("tool_call_id"),
        name=m.get("name"),
    )


def _tool_call_from_wire(tc: dict[str, Any]) -> ToolCall:
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        arguments = args
    elif isinstance(args, dict):
        arguments = json.dumps(args, ensure_ascii=False)
    else:
        arguments = ""
    return ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=arguments)


def _tools_from_wire(tools: list[dict] | None) -> list[ToolSpec] | None:
    if not tools:
        return None
    out: list[ToolSpec] = []
    for t in tools:
        if t.get("type") == "function":
            f = t.get("function") or {}
            out.append(ToolSpec(name=f.get("name", ""), description=f.get("description", ""), parameters=f.get("parameters") or {}))
        else:
            # Other tool types (e.g. "computer-use") — pass through name only.
            out.append(ToolSpec(name=t.get("type", ""), description="", parameters={}))
    return out or None


# ---------------------------------------------------------------------------
# Inbound: upstream response → NormalizedResponse (used by Mode 1)
# ---------------------------------------------------------------------------


def parse_upstream_response(body: dict[str, Any]) -> NormalizedResponse:
    choices = body.get("choices") or [{}]
    choice = choices[0]
    msg = choice.get("message") or {}

    parsed_msg = _msg_from_wire({"role": "assistant", **msg})
    finish_reason = _normalize_finish_reason(choice.get("finish_reason"))

    usage_raw = body.get("usage") or {}
    usage = Usage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
    )
    return NormalizedResponse(
        content=parsed_msg.content if isinstance(parsed_msg.content, str) else "",
        reasoning=parsed_msg.reasoning,
        tool_calls=parsed_msg.tool_calls or [],
        finish_reason=finish_reason,
        usage=usage,
    )


def parse_upstream_stream(chunks: list[dict[str, Any]]) -> NormalizedResponse:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage_raw: dict[str, Any] = {}

    for chunk in chunks:
        choices = chunk.get("choices") or []
        if choices:
            ch = choices[0]
            delta = ch.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            r = delta.get("reasoning") or delta.get("reasoning_content")
            if isinstance(r, str):
                reasoning_parts.append(r)
            for tc_delta in delta.get("tool_calls") or []:
                idx = tc_delta.get("index", 0)
                acc = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc_delta.get("id"):
                    acc["id"] = tc_delta["id"]
                fn = tc_delta.get("function") or {}
                if fn.get("name"):
                    acc["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    acc["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
        if chunk.get("usage"):
            usage_raw = chunk["usage"]

    tool_calls: list[ToolCall] = []
    for idx in sorted(tool_calls_acc):
        a = tool_calls_acc[idx]
        tool_calls.append(ToolCall(id=a["id"], name=a["name"], arguments=a["arguments"]))

    return NormalizedResponse(
        content="".join(content_parts),
        reasoning="".join(reasoning_parts) or None,
        tool_calls=tool_calls,
        finish_reason=_normalize_finish_reason(finish_reason),
        usage=Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        ),
    )


def _normalize_finish_reason(fr: str | None) -> str:
    if fr in (None, ""):
        return "stop"
    return fr


# ---------------------------------------------------------------------------
# Outbound: NormalizedResponse → wire (used by Mode 2)
# ---------------------------------------------------------------------------


def from_normalized_response_nonstream(resp: NormalizedResponse, model: str) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": resp.content or None}
    if resp.reasoning:
        msg["reasoning"] = resp.reasoning
    if resp.tool_calls:
        msg["tool_calls"] = [_tool_call_to_wire(tc) for tc in resp.tool_calls]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": _wire_finish_reason(resp),
            }
        ],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.prompt_tokens + resp.usage.completion_tokens,
        },
    }


def _wire_finish_reason(resp: NormalizedResponse) -> str:
    """OpenAI requires finish_reason='tool_calls' when tool_calls are present.
    The adapter may legitimately set 'stop' (model emitted a stop token after
    the call); patch it here so the wire response is spec-compliant."""
    if resp.tool_calls and resp.finish_reason == "stop":
        return "tool_calls"
    return resp.finish_reason


def _tool_call_to_wire(tc: ToolCall) -> dict[str, Any]:
    return {
        "id": tc.id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": tc.name, "arguments": tc.arguments},
    }


async def from_normalized_response_stream(resp: NormalizedResponse, model: str) -> AsyncIterator[str]:
    """Fake-stream a complete response as SSE chunks.

    Sequence: role chunk → content chunk → finish chunk → [DONE].
    Tool calls and reasoning ride on the content chunk's delta.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        body = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    # Role
    yield chunk({"role": "assistant", "content": ""})

    # Content / reasoning / tool_calls in one delta
    content_delta: dict[str, Any] = {}
    if resp.content:
        content_delta["content"] = resp.content
    if resp.reasoning:
        content_delta["reasoning"] = resp.reasoning
    if resp.tool_calls:
        content_delta["tool_calls"] = [{"index": i, **_tool_call_to_wire(tc)} for i, tc in enumerate(resp.tool_calls)]
    if content_delta:
        yield chunk(content_delta)

    # Finish
    yield chunk({}, finish_reason=_wire_finish_reason(resp))

    # Final usage chunk (OpenAI emits this when stream_options.include_usage=true)
    usage_body = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.prompt_tokens + resp.usage.completion_tokens,
        },
    }
    yield f"data: {json.dumps(usage_body, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
