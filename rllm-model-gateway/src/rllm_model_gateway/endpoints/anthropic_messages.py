"""Anthropic /v1/messages wire format ↔ normalized.

system top-level becomes a system Message. content blocks (text / thinking /
tool_use / tool_result) collapse to plain str + reasoning + tool_calls + tool
messages. Tool schema renames input_schema↔parameters, tool_use input dict ↔
arguments dict (with raw JSON in arguments_raw). stop_reason maps to OpenAI's
finish_reason vocabulary.
"""

from __future__ import annotations

import json
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

NAME = "anthropic_messages"
PATH = "/v1/messages"  # Anthropic SDK posts here when base_url has no /v1 suffix
UPSTREAM_PATH = "/v1/messages"  # Anthropic upstream convention: base_url has no /v1, SDK adds it

_NON_SAMPLING_KEYS = frozenset(
    {
        "model",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "stream",
        "metadata",
        "anthropic_beta",
        "anthropic_version",
    }
)


# ---------------------------------------------------------------------------
# Inbound: Anthropic request → NormalizedRequest
# ---------------------------------------------------------------------------


def to_normalized_request(body: dict[str, Any]) -> NormalizedRequest:
    messages: list[Message] = []

    # System message: top-level string or list of {type:text,text:...} blocks.
    system = body.get("system")
    if system:
        if isinstance(system, str):
            sys_text = system
        else:
            sys_text = "".join(b.get("text", "") for b in system if b.get("type") == "text")
        if sys_text:
            messages.append(Message(role="system", content=sys_text))

    for m in body.get("messages") or []:
        messages.extend(_anthropic_message_to_normalized(m))

    tools = _anthropic_tools_to_normalized(body.get("tools"))

    sampling = {k: v for k, v in body.items() if k not in _NON_SAMPLING_KEYS}
    return NormalizedRequest(
        messages=messages or None,
        tools=tools,
        sampling_params=sampling,
    )


def _anthropic_message_to_normalized(m: dict[str, Any]) -> list[Message]:
    """One Anthropic message → one or more normalized messages.

    A user message containing tool_result blocks splits into N tool messages.
    An assistant message with tool_use blocks becomes one assistant message
    with tool_calls.
    """
    role = m.get("role", "user")
    content = m.get("content")
    if isinstance(content, str):
        return [Message(role=role, content=content)]
    if not isinstance(content, list):
        return [Message(role=role, content="")]

    # Split content blocks by type.
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_results: list[Message] = []

    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            input_dict = block.get("input") or {}
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=input_dict,
                    arguments_raw=json.dumps(input_dict, ensure_ascii=False),
                )
            )
        elif btype == "tool_result":
            tr_content = block.get("content")
            if isinstance(tr_content, list):
                tr_content = "".join(c.get("text", "") for c in tr_content if c.get("type") == "text")
            tool_results.append(
                Message(
                    role="tool",
                    content=tr_content if isinstance(tr_content, str) else "",
                    tool_call_id=block.get("tool_use_id", ""),
                )
            )
        # other block types (image, document) — pass through as raw dict content
        else:
            text_parts.append(json.dumps(block, ensure_ascii=False))

    out: list[Message] = []
    if role == "user" and tool_results:
        # User turn with only tool_result blocks → emit each as a tool message.
        out.extend(tool_results)
        if text_parts:
            out.append(Message(role="user", content="\n".join(text_parts)))
        return out

    msg = Message(
        role=role,
        content="\n".join(text_parts) if text_parts else "",
        reasoning="\n".join(thinking_parts) or None,
        tool_calls=tool_calls or None,
    )
    out.append(msg)
    return out


def _anthropic_tools_to_normalized(tools: list[dict] | None) -> list[ToolSpec] | None:
    if not tools:
        return None
    return [
        ToolSpec(
            name=t.get("name", ""),
            description=t.get("description", ""),
            parameters=t.get("input_schema") or {},
        )
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Inbound: Anthropic response → NormalizedResponse (Mode 1)
# ---------------------------------------------------------------------------


def parse_upstream_response(body: dict[str, Any]) -> NormalizedResponse:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in body.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            input_dict = block.get("input") or {}
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=input_dict,
                    arguments_raw=json.dumps(input_dict, ensure_ascii=False),
                )
            )

    usage_raw = body.get("usage") or {}
    return NormalizedResponse(
        content="".join(text_parts),
        reasoning="\n".join(thinking_parts) or None,
        tool_calls=tool_calls,
        finish_reason=_anthropic_stop_reason_to_normalized(body.get("stop_reason")),
        usage=Usage(
            prompt_tokens=usage_raw.get("input_tokens", 0),
            completion_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
        ),
    )


def parse_upstream_stream(chunks: list[dict[str, Any]]) -> NormalizedResponse:
    """Reassemble from typed Anthropic SSE events."""
    blocks: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage_in = 0
    usage_out = 0

    for chunk in chunks:
        ctype = chunk.get("type")
        if ctype == "message_start":
            msg = chunk.get("message") or {}
            usage_in = (msg.get("usage") or {}).get("input_tokens", 0)
        elif ctype == "content_block_start":
            idx = chunk.get("index", 0)
            cb = chunk.get("content_block") or {}
            blocks[idx] = {
                "type": cb.get("type"),
                "text": "",
                "thinking": "",
                "id": cb.get("id"),
                "name": cb.get("name"),
                "input_partial": "",
            }
        elif ctype == "content_block_delta":
            idx = chunk.get("index", 0)
            delta = chunk.get("delta") or {}
            dt = delta.get("type")
            blk = blocks.get(idx)
            if blk is None:
                continue
            if dt == "text_delta":
                blk["text"] += delta.get("text", "")
            elif dt == "thinking_delta":
                blk["thinking"] += delta.get("thinking", "")
            elif dt == "input_json_delta":
                blk["input_partial"] += delta.get("partial_json", "")
        elif ctype == "message_delta":
            d = chunk.get("delta") or {}
            if d.get("stop_reason"):
                finish_reason = d["stop_reason"]
            u = chunk.get("usage") or {}
            usage_out = u.get("output_tokens", usage_out)

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for idx in sorted(blocks):
        b = blocks[idx]
        if b["type"] == "text":
            text_parts.append(b["text"])
        elif b["type"] == "thinking":
            thinking_parts.append(b["thinking"])
        elif b["type"] == "tool_use":
            try:
                arguments = json.loads(b["input_partial"]) if b["input_partial"] else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=b["id"] or "",
                    name=b["name"] or "",
                    arguments=arguments,
                    arguments_raw=b["input_partial"] or None,
                )
            )

    return NormalizedResponse(
        content="".join(text_parts),
        reasoning="\n".join(thinking_parts) or None,
        tool_calls=tool_calls,
        finish_reason=_anthropic_stop_reason_to_normalized(finish_reason),
        usage=Usage(prompt_tokens=usage_in, completion_tokens=usage_out, total_tokens=usage_in + usage_out),
    )


def _anthropic_stop_reason_to_normalized(sr: str | None) -> str:
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
        None: "stop",
    }.get(sr, sr or "stop")


def _normalized_finish_to_anthropic(fr: str) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "error": "end_turn",
    }.get(fr, "end_turn")


# ---------------------------------------------------------------------------
# Outbound: NormalizedResponse → Anthropic wire (Mode 2)
# ---------------------------------------------------------------------------


def from_normalized_response_nonstream(resp: NormalizedResponse, model: str) -> dict[str, Any]:
    content_blocks: list[dict[str, Any]] = []
    if resp.reasoning:
        content_blocks.append({"type": "thinking", "thinking": resp.reasoning})
    if resp.content:
        content_blocks.append({"type": "text", "text": resp.content})
    for tc in resp.tool_calls:
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.id or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tc.name,
                "input": tc.arguments,
            }
        )

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _normalized_finish_to_anthropic(resp.finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
    }


async def from_normalized_response_stream(resp: NormalizedResponse, model: str) -> AsyncIterator[str]:
    """Fake-stream Anthropic typed events from a complete NormalizedResponse."""
    msg_id = f"msg_{uuid.uuid4().hex}"

    def event(name: str, data: dict[str, Any]) -> str:
        return f"event: {name}\ndata: {json.dumps({'type': name, **data}, ensure_ascii=False)}\n\n"

    # message_start
    yield event(
        "message_start",
        {
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": 0,
                },
            }
        },
    )

    idx = 0

    # thinking block (if reasoning present)
    if resp.reasoning:
        yield event("content_block_start", {"index": idx, "content_block": {"type": "thinking", "thinking": ""}})
        yield event("content_block_delta", {"index": idx, "delta": {"type": "thinking_delta", "thinking": resp.reasoning}})
        yield event("content_block_stop", {"index": idx})
        idx += 1

    # text block (if content present)
    if resp.content:
        yield event("content_block_start", {"index": idx, "content_block": {"type": "text", "text": ""}})
        yield event("content_block_delta", {"index": idx, "delta": {"type": "text_delta", "text": resp.content}})
        yield event("content_block_stop", {"index": idx})
        idx += 1

    # tool_use blocks
    for tc in resp.tool_calls:
        yield event(
            "content_block_start",
            {
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tc.id or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": tc.name,
                    "input": {},
                },
            },
        )
        partial = tc.arguments_raw if tc.arguments_raw is not None else json.dumps(tc.arguments, ensure_ascii=False)
        yield event(
            "content_block_delta",
            {"index": idx, "delta": {"type": "input_json_delta", "partial_json": partial}},
        )
        yield event("content_block_stop", {"index": idx})
        idx += 1

    # message_delta — final stop reason + output_tokens
    yield event(
        "message_delta",
        {
            "delta": {"stop_reason": _normalized_finish_to_anthropic(resp.finish_reason), "stop_sequence": None},
            "usage": {"output_tokens": resp.usage.completion_tokens},
        },
    )

    # message_stop
    yield event("message_stop", {})
