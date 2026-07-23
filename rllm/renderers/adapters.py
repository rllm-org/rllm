"""Adapters that wrap other renderer ecosystems into the canonical interface.

- ``TinkerRendererAdapter`` wraps any ``tinker_cookbook.renderers.Renderer`` —
  which is the base class of the Fireworks cookbook renderers (e.g.
  ``DeepseekV4Renderer``), so this one adapter covers both. It supplies
  ``render_ids`` / ``parse_response`` / ``get_stop_token_ids`` from the tinker
  primitives and inherits the synthesized bridge.
- ``ChatTemplateAdapter`` wraps a HF tokenizer's chat template as a universal
  fallback (template parity not guaranteed; the bridge prefix-check protects).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .bridging import BridgingRendererMixin
from .types import ParsedResponse, RenderedTokens


def _split_content_parts(content: Any) -> tuple[str, str]:
    """Flatten OpenAI-style assistant content parts into text + reasoning.

    Fireworks cookbook reasoning renderers (including ``glm_moe_dsa``) return
    ``message.content`` as ``[{"type": "thinking", ...}, {"type": "text",
    ...}]``. The canonical renderer contract requires strings: leaking that
    list reaches the OpenAI wire response and breaks both LiteLLM and CLI
    harness schemas before they can execute a tool.
    """
    if not isinstance(content, list):
        return (content or "", "")

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            part_type = part.get("type")
            if part_type == "thinking" and part.get("thinking"):
                thinking_parts.append(str(part["thinking"]))
            elif part_type in ("text", "output_text") and part.get("text"):
                text_parts.append(str(part["text"]))
    return "\n".join(text_parts), "\n".join(thinking_parts)


def _to_parsed(content: Any, reasoning: str | None, tool_calls: Any) -> ParsedResponse:
    text, parts_reasoning = _split_content_parts(content)
    return ParsedResponse(
        content=text,
        reasoning_content=reasoning or parts_reasoning or None,
        tool_calls=list(tool_calls) if tool_calls else None,
    )


def _to_tinker_tool_call(tool_call: Any) -> Any:
    """Convert an OpenAI/rLLM tool call into cookbook's typed ToolCall.

    The canonical renderer accepts OpenAI message dictionaries, while
    tinker-cookbook renderers expect ``ToolCall`` objects in historical
    assistant messages. CLI clients send those historical calls back as
    dictionaries on turn two, so leaving them untyped breaks renderers that
    access ``tool_call.function`` (including GLM-5).
    """
    from tinker_cookbook.renderers.base import ToolCall

    if isinstance(tool_call, ToolCall):
        return tool_call
    if hasattr(tool_call, "model_dump"):
        payload = tool_call.model_dump()
    elif isinstance(tool_call, Mapping):
        payload = dict(tool_call)
    else:
        payload = {
            "id": getattr(tool_call, "id", None),
            "name": getattr(tool_call, "name", ""),
            "arguments": getattr(tool_call, "arguments", {}),
        }

    function = payload.get("function")
    if isinstance(function, Mapping):
        function = dict(function)
    else:
        function = {
            "name": payload.pop("name", ""),
            "arguments": payload.pop("arguments", {}),
        }
    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    function["arguments"] = arguments

    normalized = {
        "type": payload.get("type", "function"),
        "id": payload.get("id"),
        "function": function,
    }
    return ToolCall.model_validate(normalized)


def _to_tinker_messages(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for message in messages:
        converted = dict(message)
        tool_calls = converted.get("tool_calls")
        if tool_calls:
            converted["tool_calls"] = [_to_tinker_tool_call(call) for call in tool_calls]
        normalized.append(converted)
    return normalized


class TinkerRendererAdapter(BridgingRendererMixin):
    """Wrap a tinker-style renderer (tinker_cookbook / Fireworks cookbook)."""

    def __init__(self, inner: Any, *, close_token_ids: set[int] | None = None, synthesize_close: int | None = None):
        self._inner = inner
        stops = [int(t) for t in inner.get_stop_sequences()]
        self.close_token_ids = set(close_token_ids) if close_token_ids is not None else set(stops)
        self.synthesize_close = synthesize_close if synthesize_close is not None else (stops[0] if stops else None)

    def _with_tools(self, messages: list[dict], tools) -> list[dict]:
        if not tools:
            return list(messages)
        msgs = list(messages)
        system = ""
        if msgs and msgs[0].get("role") == "system":
            system = msgs[0].get("content") or ""
            msgs = msgs[1:]
        prefix = self._inner.create_conversation_prefix_with_tools(tools, system_prompt=system)
        return list(prefix) + msgs

    def render_ids(self, messages, *, tools=None, add_generation_prompt: bool = False) -> list[int]:
        msgs = self._with_tools(_to_tinker_messages(list(messages)), tools)
        if add_generation_prompt:
            model_input = self._inner.build_generation_prompt(msgs)
        else:
            # Closed conversation (no trailing generation prompt); requires a
            # final assistant turn — satisfied by every internal caller.
            model_input = self._inner.build_supervised_example(msgs)[0]
        return list(model_input.to_ints())

    def render(self, messages, *, tools=None, add_generation_prompt: bool = False) -> RenderedTokens:
        return RenderedTokens(token_ids=self.render_ids(messages, tools=tools, add_generation_prompt=add_generation_prompt))

    def get_stop_token_ids(self) -> list[int]:
        return [int(t) for t in self._inner.get_stop_sequences()]

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        msg, _term = self._inner.parse_response(list(token_ids))
        get = msg.get if isinstance(msg, dict) else (lambda k, d=None: getattr(msg, k, d))
        return _to_parsed(get("content", ""), get("reasoning_content"), get("tool_calls"))


class ChatTemplateAdapter(BridgingRendererMixin):
    """Universal fallback over a HF tokenizer's chat template."""

    def __init__(self, tokenizer: Any, *, close_token_ids: set[int] | None = None, synthesize_close: int | None = None):
        self._tok = tokenizer
        eos = getattr(tokenizer, "eos_token_id", None)
        self.close_token_ids = set(close_token_ids) if close_token_ids is not None else ({int(eos)} if eos is not None else set())
        self.synthesize_close = synthesize_close if synthesize_close is not None else (int(eos) if eos is not None else None)

    def render_ids(self, messages, *, tools=None, add_generation_prompt: bool = False) -> list[int]:
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": add_generation_prompt}
        if tools:
            kwargs["tools"] = tools
        return list(self._tok.apply_chat_template(list(messages), **kwargs))

    def render(self, messages, *, tools=None, add_generation_prompt: bool = False) -> RenderedTokens:
        return RenderedTokens(token_ids=self.render_ids(messages, tools=tools, add_generation_prompt=add_generation_prompt))

    def get_stop_token_ids(self) -> list[int]:
        return list(self.close_token_ids)

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        return _to_parsed(self._tok.decode(list(token_ids), skip_special_tokens=True), None, None)
