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
from typing import Any

from .bridging import BridgingRendererMixin
from .types import ParsedResponse, RenderedTokens


def _flatten_parts(content: Any) -> tuple[str, str]:
    """Split structured renderer content into visible text and thinking."""
    if not isinstance(content, list):
        return content or "", ""
    text = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    thinking = "".join(part.get("thinking", "") for part in content if isinstance(part, dict) and part.get("type") == "thinking")
    return text, thinking


def _to_openai_tool_call(tool_call: Any) -> dict[str, Any]:
    """Normalize renderer-owned tool calls to the canonical nested shape."""
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            return {
                "id": tool_call.get("id"),
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", "{}"),
                },
            }
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", "{}")
        tool_call_id = tool_call.get("id")
    else:
        function = getattr(tool_call, "function", None)
        source = function if function is not None else tool_call
        name = getattr(source, "name", "")
        arguments = getattr(source, "arguments", "{}")
        tool_call_id = getattr(tool_call, "id", None)
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _to_versioned_tool_call(tool_call: Any) -> Any:
    """Return the tool-call type required by the installed renderer API."""
    normalized = _to_openai_tool_call(tool_call)
    try:
        from renderers import ParsedToolCall  # type: ignore
    except ImportError:
        return normalized
    if isinstance(tool_call, ParsedToolCall):
        return tool_call
    function = normalized["function"]
    return ParsedToolCall(
        raw=json.dumps(normalized, separators=(",", ":")),
        name=function["name"],
        arguments=function["arguments"],
        id=normalized["id"],
    )


def _to_parsed(content: Any, reasoning: str | None, tool_calls: Any) -> ParsedResponse:
    if isinstance(content, list):
        content, thinking = _flatten_parts(content)
        reasoning = reasoning or thinking
    return ParsedResponse(
        content=content or "",
        reasoning_content=reasoning or None,
        tool_calls=[_to_versioned_tool_call(tool_call) for tool_call in tool_calls or []],
    )


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
        msgs = self._with_tools(list(messages), tools)
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

    def parse_response(self, token_ids: list[int], *, tools=None) -> ParsedResponse:
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

    def parse_response(self, token_ids: list[int], *, tools=None) -> ParsedResponse:
        return _to_parsed(self._tok.decode(list(token_ids), skip_special_tokens=True), None, None)
