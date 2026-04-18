"""OpenAI /v1/responses wire format ↔ normalized.

Heterogeneous ``input`` items (message, function_call, function_call_output,
reasoning) flatten to chat-shaped messages. Output reshapes back into
typed OutputItem list (reasoning, message, function_call).

Unsupported request features return 400: previous_response_id, background,
include filters, store. MCP / harmony / file_search / web_search / computer
tool types are silently skipped.
"""

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

NAME = "responses"
PATH = "/v1/responses"
UPSTREAM_PATH = "/responses"

# Sampling-param keys we do not forward to the adapter.
_NON_SAMPLING_KEYS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "stream",
        "metadata",
        "previous_response_id",
        "background",
        "store",
        "include",
        "user",
        "reasoning",
        "text",
        "truncation",
    }
)


# ---------------------------------------------------------------------------
# Inbound: ResponsesRequest → NormalizedRequest
# ---------------------------------------------------------------------------


def to_normalized_request(body: dict[str, Any]) -> NormalizedRequest:
    if body.get("previous_response_id"):
        raise ValueError("previous_response_id is not supported (stateful chaining deferred)")
    if body.get("background"):
        raise ValueError("background mode is not supported")

    messages: list[Message] = []
    instructions = body.get("instructions")
    if instructions:
        messages.append(Message(role="system", content=instructions))

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append(Message(role="user", content=raw_input))
    elif isinstance(raw_input, list):
        messages.extend(_flatten_input_items(raw_input))

    tools = _responses_tools_to_normalized(body.get("tools"))

    sampling = {k: v for k, v in body.items() if k not in _NON_SAMPLING_KEYS}
    # Translate `max_output_tokens` → `max_tokens` (closer to chat-completions).
    if "max_output_tokens" in sampling:
        sampling["max_tokens"] = sampling.pop("max_output_tokens")

    reasoning_cfg = body.get("reasoning") or {}
    reasoning_effort = reasoning_cfg.get("effort") if isinstance(reasoning_cfg, dict) else None

    return NormalizedRequest(
        messages=messages or None,
        tools=tools,
        sampling_params=sampling,
        reasoning_effort=reasoning_effort,
    )


def _flatten_input_items(items: list[dict[str, Any]]) -> list[Message]:
    """Convert Responses input items to a flat list of chat messages.

    Adapted from vllm/entrypoints/openai/responses/utils.py:
    construct_input_messages + _construct_single_message_from_response_item.
    Handles message, function_call, function_call_output, reasoning.
    """
    out: list[Message] = []
    for item in items:
        # Plain dicts; we don't validate against openai.types.responses to
        # avoid coupling and keep accepting partial / unknown variants.
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "message")

        if itype == "message":
            role = item.get("role", "user")
            content = item.get("content")
            text = _extract_text_from_message_content(content)
            out.append(Message(role=role, content=text))

        elif itype == "function_call":
            arguments_raw = item.get("arguments", "")
            try:
                arguments = json.loads(arguments_raw) if arguments_raw else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            out.append(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=item.get("call_id") or item.get("id", ""),
                            name=item.get("name", ""),
                            arguments=arguments,
                            arguments_raw=arguments_raw if isinstance(arguments_raw, str) else None,
                        )
                    ],
                )
            )

        elif itype == "function_call_output":
            out.append(
                Message(
                    role="tool",
                    content=item.get("output", ""),
                    tool_call_id=item.get("call_id", ""),
                )
            )

        elif itype == "reasoning":
            text = ""
            summary = item.get("summary") or []
            if isinstance(summary, list) and summary:
                text = summary[0].get("text", "")
            elif item.get("content"):
                content = item["content"]
                if isinstance(content, list) and content:
                    text = content[0].get("text", "")
            if text:
                out.append(Message(role="assistant", content="", reasoning=text))

        # Unhandled types (computer_call, file_search_call, mcp_call, etc.):
        # ignore. Could log; not raising to remain forgiving on Mode 1 traces.
    return out


def _extract_text_from_message_content(content: Any) -> str:
    """Pull plain text from a Responses message content field.

    content may be a str, or a list of {type: "input_text"|"output_text"|..., text: ...}.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type", "")
            if t in ("input_text", "output_text", "text"):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


def _responses_tools_to_normalized(tools: list[dict] | None) -> list[ToolSpec] | None:
    """Responses tools have a flat shape: {type: "function", name, description, parameters}.

    Other tool types (mcp, computer_use, file_search, web_search, code_interpreter)
    require server-side execution and aren't supported. They're skipped.
    """
    if not tools:
        return None
    out: list[ToolSpec] = []
    for t in tools:
        if t.get("type") == "function":
            out.append(
                ToolSpec(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("parameters") or {},
                )
            )
    return out or None


# ---------------------------------------------------------------------------
# Inbound: ResponsesResponse → NormalizedResponse (Mode 1)
# ---------------------------------------------------------------------------


def parse_upstream_response(body: dict[str, Any]) -> NormalizedResponse:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            text_parts.append(_extract_text_from_message_content(item.get("content")))
        elif itype == "reasoning":
            summary = item.get("summary") or []
            if isinstance(summary, list) and summary:
                reasoning_parts.append(summary[0].get("text", ""))
            elif item.get("content"):
                content = item["content"]
                if isinstance(content, list) and content:
                    reasoning_parts.append(content[0].get("text", ""))
        elif itype == "function_call":
            arguments_raw = item.get("arguments", "")
            try:
                arguments = json.loads(arguments_raw) if arguments_raw else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=item.get("call_id") or item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=arguments,
                    arguments_raw=arguments_raw if isinstance(arguments_raw, str) else None,
                )
            )

    usage_raw = body.get("usage") or {}
    finish_reason = _responses_status_to_finish_reason(body.get("status"), tool_calls)

    return NormalizedResponse(
        content="".join(text_parts),
        reasoning="\n".join(reasoning_parts) or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=Usage(
            prompt_tokens=usage_raw.get("input_tokens", 0),
            completion_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
        ),
    )


def parse_upstream_stream(chunks: list[dict[str, Any]]) -> NormalizedResponse:
    """Reassemble a NormalizedResponse from a Responses SSE event stream.

    The terminal ``response.completed`` event carries the full response
    object; if present we just parse that. Otherwise we accumulate deltas.
    """
    final = None
    for chunk in chunks:
        ctype = chunk.get("type")
        if ctype == "response.completed":
            final = chunk.get("response")
            break

    if final is not None:
        return parse_upstream_response(final)

    # Fallback: accumulate from deltas (defensive; upstream usually emits completed).
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        ctype = chunk.get("type")
        if ctype == "response.output_text.delta":
            text_parts.append(chunk.get("delta", ""))
        elif ctype == "response.reasoning_text.delta":
            reasoning_parts.append(chunk.get("delta", ""))
        elif ctype == "response.function_call_arguments.delta":
            iid = chunk.get("item_id", "")
            tc = tool_calls_acc.setdefault(iid, {"id": iid, "name": "", "arguments_raw": ""})
            tc["arguments_raw"] += chunk.get("delta", "")
        elif ctype == "response.output_item.added":
            item = chunk.get("item") or {}
            if item.get("type") == "function_call":
                iid = item.get("id", "")
                tc = tool_calls_acc.setdefault(iid, {"id": iid, "name": "", "arguments_raw": ""})
                tc["name"] = item.get("name", "")
                tc["call_id"] = item.get("call_id", "")

    tool_calls: list[ToolCall] = []
    for iid in tool_calls_acc:
        a = tool_calls_acc[iid]
        try:
            arguments = json.loads(a["arguments_raw"]) if a["arguments_raw"] else {}
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        tool_calls.append(
            ToolCall(
                id=a.get("call_id") or a["id"] or "",
                name=a["name"],
                arguments=arguments,
                arguments_raw=a["arguments_raw"] or None,
            )
        )

    return NormalizedResponse(
        content="".join(text_parts),
        reasoning="".join(reasoning_parts) or None,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
    )


def _responses_status_to_finish_reason(status: str | None, tool_calls: list) -> str:
    if status == "incomplete":
        return "length"
    if tool_calls:
        return "tool_calls"
    return "stop"


# ---------------------------------------------------------------------------
# Outbound: NormalizedResponse → ResponsesResponse (Mode 2)
# ---------------------------------------------------------------------------


def from_normalized_response_nonstream(resp: NormalizedResponse, model: str) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    output_items: list[dict[str, Any]] = []

    if resp.reasoning:
        output_items.append(
            {
                "id": f"rs_{uuid.uuid4().hex}",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": resp.reasoning}],
            }
        )

    if resp.content:
        output_items.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": resp.content, "annotations": []}],
            }
        )

    for tc in resp.tool_calls:
        output_items.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": tc.id or f"call_{uuid.uuid4().hex[:24]}",
                "name": tc.name,
                "arguments": tc.arguments_raw if tc.arguments_raw is not None else json.dumps(tc.arguments, ensure_ascii=False),
            }
        )

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output_items,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.prompt_tokens + resp.usage.completion_tokens,
        },
        "metadata": {},
        "parallel_tool_calls": True,
        "tool_choice": "auto",
    }


async def from_normalized_response_stream(resp: NormalizedResponse, model: str) -> AsyncIterator[str]:
    """Fake-stream Responses typed events.

    Sequence:
      response.created
      response.in_progress
      [reasoning item, if any]
        response.output_item.added
        response.reasoning_summary_part.added (one summary part)
        response.reasoning_summary_text.delta
        response.reasoning_summary_text.done
        response.reasoning_summary_part.done
        response.output_item.done
      [message item, if content]
        response.output_item.added
        response.content_part.added
        response.output_text.delta
        response.output_text.done
        response.content_part.done
        response.output_item.done
      [for each tool call]
        response.output_item.added
        response.function_call_arguments.delta
        response.function_call_arguments.done
        response.output_item.done
      response.completed
    """
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())

    base_response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "model": model,
        "status": "in_progress",
        "output": [],
        "usage": None,
        "metadata": {},
    }

    def event(name: str, payload: dict[str, Any]) -> str:
        body = {"type": name, **payload}
        return f"event: {name}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

    # response.created / in_progress
    yield event("response.created", {"response": dict(base_response)})
    yield event("response.in_progress", {"response": dict(base_response)})

    output_index = 0
    output_items: list[dict[str, Any]] = []

    # Reasoning item
    if resp.reasoning:
        rs_id = f"rs_{uuid.uuid4().hex}"
        item = {
            "id": rs_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
        }
        yield event("response.output_item.added", {"output_index": output_index, "item": item})
        yield event(
            "response.reasoning_summary_part.added",
            {
                "item_id": rs_id,
                "output_index": output_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": ""},
            },
        )
        yield event(
            "response.reasoning_summary_text.delta",
            {
                "item_id": rs_id,
                "output_index": output_index,
                "summary_index": 0,
                "delta": resp.reasoning,
            },
        )
        yield event(
            "response.reasoning_summary_text.done",
            {
                "item_id": rs_id,
                "output_index": output_index,
                "summary_index": 0,
                "text": resp.reasoning,
            },
        )
        yield event(
            "response.reasoning_summary_part.done",
            {
                "item_id": rs_id,
                "output_index": output_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": resp.reasoning},
            },
        )
        item_done = {**item, "status": "completed", "summary": [{"type": "summary_text", "text": resp.reasoning}]}
        yield event("response.output_item.done", {"output_index": output_index, "item": item_done})
        output_items.append(item_done)
        output_index += 1

    # Message item
    if resp.content:
        msg_id = f"msg_{uuid.uuid4().hex}"
        item = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        yield event("response.output_item.added", {"output_index": output_index, "item": item})
        yield event(
            "response.content_part.added",
            {
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )
        yield event(
            "response.output_text.delta",
            {
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": resp.content,
            },
        )
        yield event(
            "response.output_text.done",
            {
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": 0,
                "text": resp.content,
            },
        )
        yield event(
            "response.content_part.done",
            {
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": resp.content, "annotations": []},
            },
        )
        item_done = {
            **item,
            "status": "completed",
            "content": [{"type": "output_text", "text": resp.content, "annotations": []}],
        }
        yield event("response.output_item.done", {"output_index": output_index, "item": item_done})
        output_items.append(item_done)
        output_index += 1

    # Function call items
    for tc in resp.tool_calls:
        fc_id = f"fc_{uuid.uuid4().hex}"
        call_id = tc.id or f"call_{uuid.uuid4().hex[:24]}"
        arguments = tc.arguments_raw if tc.arguments_raw is not None else json.dumps(tc.arguments, ensure_ascii=False)
        item = {
            "id": fc_id,
            "type": "function_call",
            "status": "in_progress",
            "call_id": call_id,
            "name": tc.name,
            "arguments": "",
        }
        yield event("response.output_item.added", {"output_index": output_index, "item": item})
        yield event(
            "response.function_call_arguments.delta",
            {"item_id": fc_id, "output_index": output_index, "delta": arguments},
        )
        yield event(
            "response.function_call_arguments.done",
            {"item_id": fc_id, "output_index": output_index, "arguments": arguments},
        )
        item_done = {**item, "status": "completed", "arguments": arguments}
        yield event("response.output_item.done", {"output_index": output_index, "item": item_done})
        output_items.append(item_done)
        output_index += 1

    # response.completed
    final_response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": output_items,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.prompt_tokens + resp.usage.completion_tokens,
        },
        "metadata": {},
    }
    yield event("response.completed", {"response": final_response})
