"""Responses API 格式适配器。

这个文件解决什么问题：
  Agent（如 Codex CLI）发出 Responses API 格式的请求，但 gateway 的训练数据提取
  （data_process.py）只认 Chat Completions 格式。本模块负责两者之间的翻译。

两种 API 格式的结构对比：

  Responses API 请求:                    Chat Completions 请求:
  {                                      {
    "model": "gpt-4",                      "model": "gpt-4",
    "instructions": "你是助手",            "messages": [
    "input": "你好",                         {"role": "system", "content": "你是助手"},
    "max_output_tokens": 1024,               {"role": "user", "content": "你好"}
  }                                        ],
                                           "max_tokens": 1024,
                                         }

  Responses API 响应:                    Chat Completions 响应:
  {                                      {
    "id": "resp_xxx",                      "id": "chatcmpl-xxx",
    "object": "response",                  "object": "chat.completion",
    "output": [{                           "choices": [{
      "type": "message",                     "message": {
      "content": [{                            "role": "assistant",
        "type": "output_text",                 "content": "你好！"
        "text": "你好！"                     }
      }]                                   }],
    }],                                    "usage": {"prompt_tokens": 10, ...}
    "usage": {"input_tokens": 10, ...}   }
  }

  Responses API 流式事件:               Chat Completions 流式事件:
  event: response.created                data: {"choices":[{"delta":{"role":"assistant"}}]}
  event: response.output_text.delta      data: {"choices":[{"delta":{"content":"你"}}]}
  event: response.output_text.delta      data: {"choices":[{"delta":{"content":"好"}}]}
  event: response.completed              data: [DONE]

工作流：
  1. proxy 收到 Responses 请求 → adapter.to_chat_completion() → 变成 Chat Completions
  2. 发给 vLLM，vLLM 返回 Chat Completions 格式
  3. data_process.py 从 Chat Completions 响应提取 token_ids（训练用）
  4. adapter.from_chat_completion() → 翻译回 Responses 格式返给 agent

Reference: Dressage blackbox_server/proxy/rollout_llm_proxy.py L1631-1769
"""

from __future__ import annotations

import json
from typing import Any

# ===========================================================================
# 公开接口：ResponsesAdapter（proxy.py 只调这个类的方法）
# ===========================================================================


class ResponsesAdapter:
    """翻译器：Responses API <-> Chat Completions。

    proxy.py 在启动时根据 RLLM_API_FORMAT 环境变量决定是否激活本 adapter。
    激活后，所有进出 gateway 的请求/响应都经过本类翻译。
    """

    def to_chat_completion(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """把 agent 发来的 Responses 请求翻译成 Chat Completions 请求。

        对应 Dressage: _openai_responses_to_chat_completion() L1631-1660
        差异: rLLM 新增 images side-channel（v2）—— 从 input_image blocks 顺序化收集
              data URL 到 ctx["images"]，供 middleware 转交给 SessionManager 与
              vlm_tito 用于 vLLM multimodal 推理。

        返回值:
          - chat_body: 翻译后的 Chat Completions 格式请求体，发给 vLLM
          - ctx: 上下文字典，保存原始请求 + images data URL 列表，供后续
                 from_chat_completion() 翻译响应时以及 middleware 使用
        """
        chat_body, images = _responses_to_chat(body)
        ctx: dict[str, Any] = {"original_request": body, "images": images}
        return chat_body, ctx

    def from_chat_completion(self, response: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """把 vLLM 返回的 Chat Completions 响应翻译回 Responses 格式，返给 agent。

        对应 Dressage: _chat_completion_to_openai_response() L1662-1684
        差异: rLLM 新增错误 passthrough guard（Dressage 用独立的 error handler L1883-1968）

        如果 vLLM 返回的是错误（无 choices、有 error key），直接 passthrough 不翻译。

        示例:
          输入: {"choices": [{"message": {"content": "你好！"}}], "usage": {...}}
          输出: {"object": "response", "output": [{"type": "message", ...}], "usage": {...}}
        """
        if "error" in response and "choices" not in response:
            return response
        return _chat_to_responses(response, ctx["original_request"])

    def translate_stream_chunk(self, chunk: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
        """Sync per-chunk API for ASGI middleware. State is stored on ``ctx``.

        Middleware buffers SSE events at ``\\n\\n`` boundaries and calls this per
        parsed chunk. Returned strings are already fully formatted SSE events
        (``event: X\\ndata: Y\\n\\n``); the middleware just concatenates them.
        """
        state = ctx.setdefault("_stream_state", _new_stream_state())
        return _translate_one_chunk(chunk, state, ctx)

    def flush_stream(self, ctx: dict[str, Any]) -> list[str]:
        """Emit terminal events when the SSE stream closes.

        Called by ``_SSETranslatingSend`` on ``more_body=False`` so Codex
        receives a complete ``response.completed`` event even if the upstream
        ended without a ``[DONE]`` marker.
        """
        state = ctx.get("_stream_state") or _new_stream_state()
        return _flush_stream(state)


# ===========================================================================
# Adapter 注册表（按 RLLM_API_FORMAT 环境变量的值查找）
# ===========================================================================

ADAPTERS: dict[str, ResponsesAdapter | None] = {
    "chat": None,
    "responses": ResponsesAdapter(),
}


# ===========================================================================
# 请求翻译：Responses API 请求体 → Chat Completions 请求体
# ===========================================================================


def _responses_to_chat(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """把完整的 Responses 请求体转换为 Chat Completions 请求体。

    对应 Dressage: _openai_responses_to_chat_completion() L1631-1660 内部逻辑
    差异: rLLM 新增 images side-channel（v2）——data URL 顺序化收集，透传给
          middleware / SessionManager 供 vlm_tito 使用。

    字段映射:
      instructions      → messages[0] (role=system)
      input (str/list)  → messages (role=user/assistant/tool)
      max_output_tokens → max_tokens
      tools             → tools (格式略有不同，需要转换)
      tool_choice       → tool_choice
      temperature/top_p → 直接透传

    返回 (chat_body, images):
      - chat_body: Chat Completions 请求体
      - images: input 中 input_image 的 data URL 列表（顺序保留，可能为空）
    """
    system_parts: list[str] = []
    instructions = _content_to_text(payload.get("instructions"))
    if instructions:
        system_parts.append(instructions)

    input_system_parts, messages, images = _input_to_messages(payload.get("input"))
    system_parts.extend(input_system_parts)

    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    result: dict[str, Any] = {
        "model": payload.get("model") or "proxy-model",
        "messages": messages,
        "stream": bool(payload.get("stream", False)),
    }

    for src, tgt in (
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("parallel_tool_calls", "parallel_tool_calls"),
    ):
        if src in payload:
            result[tgt] = payload[src]

    tools = _tools_to_chat(payload.get("tools"))
    if tools:
        result["tools"] = tools
    tool_choice = _tool_choice_to_chat(payload.get("tool_choice"))
    if tool_choice is not None:
        result["tool_choice"] = tool_choice
    return result, images


# ===========================================================================
# 响应翻译：Chat Completions 响应体 → Responses API 响应体
# ===========================================================================


def _chat_to_responses(payload: dict[str, Any], original_request: dict[str, Any]) -> dict[str, Any]:
    """把 vLLM 返回的 Chat Completions 响应翻译为 Responses API 响应。

    对应 Dressage: _chat_completion_to_openai_response() L1662-1684
    差异: 无，1:1

    字段映射:
      choices[0].message.content    → output[].type="message" (含 output_text)
      choices[0].message.tool_calls → output[].type="function_call"
      usage.prompt_tokens           → usage.input_tokens
      usage.completion_tokens       → usage.output_tokens
    """
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}

    output: list[dict[str, Any]] = []

    text = _content_to_text(message.get("content"))
    if text or not isinstance(message.get("tool_calls"), list):
        output.append(_message_item(text))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        output.extend(_tool_calls_to_response_items(tool_calls))

    return {
        "id": str(payload.get("id") or "resp_proxy"),
        "object": "response",
        "status": "completed",
        "model": str(payload.get("model") or original_request.get("model") or "proxy-model"),
        "output": output,
        "usage": _usage_to_responses(payload.get("usage")),
    }


# ===========================================================================
# 流式事件构造 helpers
#
# ``ResponsesAdapter.translate_stream_chunk`` (sync-per-chunk, ASGI middleware)
# delegates here so the state machine lives in one place.
# ===========================================================================


def _new_stream_state() -> dict[str, Any]:
    """Fresh mutable state for cross-chunk accumulation.

    Fields:
      preamble_sent — True after the 3 preamble events have been emitted
      text_parts    — pieces of assistant text seen so far (joined at flush)
      tool_call_buffers — index -> {id, name, arguments} accumulators
      usage         — Responses-style usage dict; last non-None wins
      response_id   — assigned on the first chunk seen (from id() of ctx)
      item_id       — fixed for a single response; part of every event
      model         — last non-empty ``model`` from chunk or ctx original_request
    """
    return {
        "preamble_sent": False,
        "text_parts": [],
        "tool_call_buffers": {},
        "usage": None,
        "response_id": None,
        "item_id": "msg_proxy_stream",
        "model": "proxy-model",
    }


def _translate_one_chunk(
    chunk: dict[str, Any],
    state: dict[str, Any],
    ctx: dict[str, Any],
) -> list[str]:
    """Translate one Chat Completions chunk into 0-N Responses SSE event strings.

    Mutates ``state`` for cross-chunk accumulation (text parts, tool-call
    buffers, preamble-sent flag). Appends the raw chunk to ``ctx["chunks"]``
    so callers (proxy, middleware) can build trace records afterwards.

    Returns already fully-formatted SSE event strings (``event: X\\ndata: Y\\n\\n``);
    callers just concatenate.
    """
    events: list[str] = []
    ctx.setdefault("chunks", []).append(chunk)

    if state["response_id"] is None:
        state["response_id"] = f"resp_{id(ctx):x}"

    if chunk.get("model"):
        state["model"] = str(chunk["model"])
    elif ctx.get("original_request", {}).get("model"):
        state["model"] = str(ctx["original_request"]["model"])

    if isinstance(chunk.get("usage"), dict):
        state["usage"] = _usage_to_responses(chunk["usage"])

    if not state["preamble_sent"]:
        state["preamble_sent"] = True
        events.extend(_preamble_events(state["response_id"], state["model"], state["item_id"]))

    choices = chunk.get("choices") or []
    if not choices:
        return events
    delta = choices[0].get("delta") or {}

    content = delta.get("content")
    if content:
        state["text_parts"].append(str(content))
        events.append(_text_delta_event(state["item_id"], str(content)))

    delta_tool_calls = delta.get("tool_calls")
    if isinstance(delta_tool_calls, list):
        for tc_delta in delta_tool_calls:
            if not isinstance(tc_delta, dict):
                continue
            idx = int(tc_delta.get("index", 0))
            buf = state["tool_call_buffers"].setdefault(
                idx, {"id": "", "name": "", "arguments": ""}
            )
            if tc_delta.get("id"):
                buf["id"] = str(tc_delta["id"])
            fn = tc_delta.get("function") or {}
            if isinstance(fn, dict):
                if fn.get("name"):
                    buf["name"] += str(fn["name"])
                if fn.get("arguments"):
                    buf["arguments"] += str(fn["arguments"])

    return events


def _flush_stream(state: dict[str, Any]) -> list[str]:
    """Emit terminal SSE events based on accumulated state.

    Called after all chunks are processed (or on stream close from the
    middleware side). Builds the ``response.output_text.done``,
    ``content_part.done``, ``output_item.done``, and ``response.completed``
    events, plus per-tool-call ``output_item.added/done`` events.
    """
    final_text = "".join(state["text_parts"])
    tool_call_items: list[dict[str, Any]] = []
    for idx in sorted(state["tool_call_buffers"].keys()):
        buf = state["tool_call_buffers"][idx]
        call_id = buf["id"] or f"call_proxy_{idx}"
        tool_call_items.append(
            {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": buf["name"] or "tool",
                "arguments": buf["arguments"],
                "status": "completed",
            }
        )
    return _completion_events(
        state["response_id"] or "resp_flush",
        state["model"],
        state["item_id"],
        final_text,
        tool_call_items,
        state["usage"],
    )


def _preamble_events(response_id: str, model: str, item_id: str) -> list[str]:
    """流开始时发送的 3 个事件（告知 agent："响应开始了，准备接收文本"）。

    对应 Dressage: _iter_openai_response_events_from_chat_stream() 顶部 L1698-1730
    差异: Dressage 内联；rLLM 抽为独立 helper
    """
    return [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                    "usage": None,
                },
            },
        ),
        _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": _message_item("", status="in_progress", item_id=item_id),
            },
        ),
        _sse(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
    ]


def _text_delta_event(item_id: str, content: str) -> str:
    """一个文本增量事件（agent 收到后追加到当前输出）。

    对应 Dressage: _iter_openai_response_events_from_chat_stream() 循环内 L1752-1762
    差异: Dressage 内联；rLLM 抽为独立 helper
    """
    return _sse(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": content,
        },
    )


def _completion_events(
    response_id: str,
    model: str,
    item_id: str,
    final_text: str,
    tool_call_items: list[dict[str, Any]],
    usage: dict[str, int] | None,
) -> list[str]:
    """流结束时发送的收尾事件。

    对应 Dressage: _openai_response_stream_completion_events() L2247-2299
    差异: Dressage 只发 text 相关事件，不含 tool_call items；rLLM 新增 tool_call 发射

    包含:
      - text.done / content_part.done / output_item.done（消息部分）
      - 每个 tool_call 的 output_item.added + output_item.done
      - response.completed（含完整 output 数组）
    """
    if usage is None:
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    events: list[str] = []

    events.append(
        _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": final_text,
            },
        )
    )
    events.append(
        _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": final_text},
            },
        )
    )
    events.append(
        _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "item_id": item_id,
                "output_index": 0,
                "item": _message_item(final_text, item_id=item_id),
            },
        )
    )

    for i, tc_item in enumerate(tool_call_items):
        output_index = i + 1
        events.append(
            _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": tc_item,
                },
            )
        )
        events.append(
            _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": tc_item,
                },
            )
        )

    output: list[dict[str, Any]] = [_message_item(final_text, item_id=item_id)]
    output.extend(tool_call_items)
    events.append(
        _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "model": model,
                    "output": output,
                    "usage": usage,
                },
            },
        )
    )
    return events


# ===========================================================================
# 基础工具函数
# ===========================================================================


def _sse(event: str, payload: dict[str, Any]) -> str:
    """构造一条 SSE 事件字符串（格式: "event: xxx\\ndata: {json}\\n\\n"）。

    对应 Dressage: _openai_response_sse() L1972-1973
    差异: Dressage 返回 bytes；rLLM 返回 str（proxy 的 StreamingResponse 接受 str）
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _message_item(text: str, *, status: str = "completed", item_id: str = "msg_proxy") -> dict[str, Any]:
    """构造 Responses API 的 message output item（agent 看到的一条消息）。

    对应 Dressage: _openai_response_message_item() L2106-2118
    差异: 无，1:1
    """
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _content_to_text(value: Any) -> str:
    """把 Responses API 中各种 content 格式统一提取为纯文本。

    对应 Dressage: _openai_responses_content_to_text()（同文件内）
    差异: 无，1:1

    Responses API 的 content 可能是:
      - None → ""
      - "直接文本" → "直接文本"
      - [{"type": "output_text", "text": "xxx"}, ...] → "xxx"
      - 其他 → JSON 序列化
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("output") is not None:
                    parts.append(str(item["output"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return json.dumps(value, ensure_ascii=False)


def _input_to_messages(value: Any) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """把 Responses API 的 "input" 字段转换为 Chat Completions 的 messages 数组。

    对应 Dressage: _openai_responses_input_to_chat_messages() L2003-2048
    差异: Dressage 每个 function_call 独立生成一条 assistant msg（有 bug）；
          rLLM 用 pending_tool_calls buffer + _flush_tool_calls() 合并相邻 function_call。
          v2 新增：input_image 的 data URL 同步收集到 images 侧通道返回。

    Responses API 的 input 可能是:
      - "简单文本" → [{"role": "user", "content": "简单文本"}]
      - [{type: "message", role: "user", content: "..."}, ...] → 逐个转换
      - [{type: "function_call", ...}] → {"role": "assistant", "tool_calls": [...]}
      - [{type: "function_call_output", ...}] → {"role": "tool", ...}
      - [{type: "input_image", image_url: "data:..."}, ...] → user 消息里的 image_url part
                                                              + images 列表收集 URL

    返回 (system_parts, messages, images):
      - system_parts: input 中 role=developer/system 的内容（合并到 system 消息）
      - messages:     其余消息的 Chat Completions 格式
      - images:       所有 input_image 的 data URL（保留顺序，可能为空）
    """
    if value is None:
        return [], [], []
    if isinstance(value, str):
        return [], [{"role": "user", "content": value}], []
    if not isinstance(value, list):
        return [], [{"role": "user", "content": _content_to_text(value)}], []

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    images: list[str] = []
    pending_tool_calls: list[dict[str, Any]] = []
    pending_image_parts: list[dict[str, Any]] = []

    def _flush_tool_calls():
        if pending_tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": list(pending_tool_calls),
                }
            )
            pending_tool_calls.clear()

    def _flush_image_parts():
        if pending_image_parts:
            messages.append({"role": "user", "content": list(pending_image_parts)})
            pending_image_parts.clear()

    def _build_user_content(raw_content: Any) -> Any:
        """Build user message content, merging with any pending image parts."""
        if not pending_image_parts:
            return _content_to_text(raw_content)
        parts = list(pending_image_parts)
        pending_image_parts.clear()
        text = _content_to_text(raw_content)
        if text:
            parts.append({"type": "text", "text": text})
        return parts

    for item in value:
        if isinstance(item, str):
            _flush_tool_calls()
            messages.append({"role": "user", "content": _build_user_content(item)})
            continue
        if not isinstance(item, dict):
            _flush_tool_calls()
            messages.append({"role": "user", "content": _build_user_content(item)})
            continue
        item_type = item.get("type")
        if item_type == "input_image":
            _flush_tool_calls()
            image_url = item.get("image_url", "")
            pending_image_parts.append({"type": "image_url", "image_url": {"url": image_url}})
            if image_url:
                images.append(image_url)
        elif item_type == "function_call":
            _flush_image_parts()
            pending_tool_calls.append(_make_tool_call_obj(item))
        elif item_type == "function_call_output":
            _flush_tool_calls()
            _flush_image_parts()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or "call_proxy"),
                    "content": _content_to_text(item.get("output")),
                }
            )
        elif item_type == "message" or item.get("role") is not None:
            _flush_tool_calls()
            role = str(item.get("role") or "user").lower()
            if role in {"developer", "system"}:
                _flush_image_parts()
                content = _content_to_text(item.get("content"))
                if content:
                    system_parts.append(content)
                continue
            if role not in {"user", "assistant", "tool"}:
                role = "user"
            if role == "user":
                content = _build_user_content(item.get("content"))
            else:
                _flush_image_parts()
                content = _content_to_text(item.get("content"))
            messages.append({"role": role, "content": content})
        else:
            _flush_tool_calls()
            _flush_image_parts()
            messages.append({"role": "user", "content": json.dumps(item, ensure_ascii=False)})

    _flush_tool_calls()
    _flush_image_parts()
    return system_parts, messages, images


def _make_tool_call_obj(item: dict[str, Any]) -> dict[str, Any]:
    """把 Responses API 的 function_call item 转为 Chat Completions 的 tool_call 对象。

    对应 Dressage: _openai_response_function_call_to_chat_message() L2051-2068
    差异: Dressage 返回完整 {role: assistant, tool_calls: [...]} 消息；
          rLLM 只返回内部 tool_call 对象，由 _flush_tool_calls() 包装为消息（修 C2）

    Responses: {"type": "function_call", "name": "calc", "arguments": "...", "call_id": "xxx"}
    返回:      {"id": "xxx", "type": "function", "function": {"name": "calc", "arguments": "..."}}
    """
    arguments = item.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "id": str(item.get("call_id") or item.get("id") or "call_proxy"),
        "type": "function",
        "function": {
            "name": str(item.get("name") or "tool"),
            "arguments": arguments,
        },
    }


def _tools_to_chat(value: Any) -> list[dict[str, Any]]:
    """把 Responses API 的 tools 列表转为 Chat Completions 格式。

    对应 Dressage: _openai_responses_tools_to_chat_tools() L2071-2089
    差异: 无，1:1

    两者差异很小：Responses 中 function 定义可能直接在 tool 对象上，
    Chat Completions 要求包在 {"type": "function", "function": {...}} 里。
    """
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for tool in value:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        raw = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(raw, dict):
            continue
        fn: dict[str, Any] = {
            "name": str(raw.get("name") or "tool"),
            "description": str(raw.get("description") or ""),
            "parameters": raw.get("parameters") or {"type": "object", "properties": {}},
        }
        if "strict" in raw:
            fn["strict"] = raw["strict"]
        tools.append({"type": "function", "function": fn})
    return tools


def _tool_choice_to_chat(value: Any) -> Any:
    """把 Responses API 的 tool_choice 转为 Chat Completions 格式。

    对应 Dressage: _openai_responses_tool_choice_to_chat() L2092-2103
    差异: 无，1:1

    Responses: {"type": "function", "function": {"name": "calc"}}  或  "auto"/"none"
    Chat:      同上（格式一致，直接透传字符串值；对象值需标准化）
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("type") == "function":
        fn = value.get("function") if isinstance(value.get("function"), dict) else value
        if isinstance(fn, dict) and fn.get("name"):
            return {"type": "function", "function": {"name": str(fn["name"])}}
    return None


def _tool_calls_to_response_items(value: Any) -> list[dict[str, Any]]:
    """把 Chat Completions 响应中的 tool_calls 转为 Responses API 的 function_call output items。

    对应 Dressage: _openai_tool_calls_to_response_items() L2143-2169
    差异: 无，1:1

    Chat:      {"id": "call_1", "function": {"name": "calc", "arguments": "{...}"}}
    Responses: {"type": "function_call", "call_id": "call_1", "name": "calc", "arguments": "{...}"}
    """
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for i, tc in enumerate(value):
        if not isinstance(tc, dict):
            continue
        function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        if not isinstance(function, dict):
            function = {}
        call_id = str(tc.get("id") or f"call_proxy_{i}")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": str(function.get("name") or "tool"),
                "arguments": arguments,
                "status": "completed",
            }
        )
    return items


def _usage_to_responses(value: Any) -> dict[str, int]:
    """把 Chat Completions 的 usage 字段名映射到 Responses API 的字段名。

    对应 Dressage: _openai_usage_to_response_usage() L2172-2182
    差异: 无，1:1

    Chat:      {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    Responses: {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    """
    if not isinstance(value, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
    output_tokens = int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0)
    total_tokens = int(value.get("total_tokens", input_tokens + output_tokens) or 0)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}
