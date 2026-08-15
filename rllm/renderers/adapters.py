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


def _render_with_prefix_attribution(renderer, messages, *, tools, add_generation_prompt) -> RenderedTokens:
    """Attribute fallback-renderer tokens without guessing ownership."""
    previous: list[int] = []
    indices: list[int] = []
    for index in range(len(messages)):
        current = renderer(messages[: index + 1], tools=tools, add_generation_prompt=False)
        if current[: len(previous)] != previous:
            raise ValueError("renderer rewrites previously rendered history, so rLLM cannot derive an exact per-message SFT loss mask; pin a native renderer")
        indices.extend([index] * (len(current) - len(previous)))
        previous = current

    if add_generation_prompt:
        current = renderer(messages, tools=tools, add_generation_prompt=True)
        if current[: len(previous)] != previous:
            raise ValueError("renderer generation prompt rewrites the closed conversation")
        indices.extend([-1] * (len(current) - len(previous)))
        previous = current
    return RenderedTokens(token_ids=previous, message_indices=indices)


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


def to_tinker_messages(messages: list[dict]) -> list[dict]:
    """Translate renderer-wire messages into cookbook message objects."""
    from tinker_cookbook.renderers.base import ToolCall

    tool_call_fields = set(ToolCall.model_fields)
    converted_messages: list[dict] = []
    for message in messages:
        converted = {key: value for key, value in message.items() if key not in {"reasoning_content", "trainable"}}
        converted["content"] = message.get("content") or ""
        reasoning = message.get("reasoning_content") if message.get("role") == "assistant" else ""
        if reasoning:
            thinking = {"type": "thinking", "thinking": reasoning}
            content = converted["content"]
            if isinstance(content, str):
                converted["content"] = [thinking, {"type": "text", "text": content}]
            elif not any(isinstance(part, dict) and part.get("type") == "thinking" for part in content):
                converted["content"] = [thinking, *content]
        tool_calls = message.get("tool_calls")
        if tool_calls:
            converted["tool_calls"] = [
                ToolCall.model_validate({key: value for key, value in tool_call.items() if key in tool_call_fields}) if isinstance(tool_call, dict) else tool_call for tool_call in tool_calls
            ]
        converted_messages.append(converted)
    return converted_messages


def to_tinker_tool_specs(tools: list[dict] | None) -> list:
    """Translate OpenAI function declarations into cookbook tool specs."""
    from tinker_cookbook.renderers.base import ToolSpec

    specs: list[ToolSpec] = []
    for index, tool in enumerate(tools or []):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"Tool declaration {index} must be an OpenAI function tool.")
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            raise ValueError(f"Tool declaration {index} needs a non-empty function.name.")
        parameters = function.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError(f"Tool declaration {index} function.parameters must be an object.")
        specs.append(
            ToolSpec(
                name=function["name"],
                description=function.get("description") or "",
                parameters=parameters,
            )
        )
    return specs


def prepare_tinker_messages_with_tools(renderer: Any, messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Inject tool declarations exactly as the cookbook renderer expects."""
    if not tools:
        return list(messages)

    remaining = list(messages)
    system_prompt = ""
    if remaining and remaining[0].get("role") == "system":
        content = remaining[0].get("content") or ""
        if isinstance(content, str):
            system_prompt = content
        elif isinstance(content, list):
            system_prompt = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        remaining = remaining[1:]

    prefix = renderer.create_conversation_prefix_with_tools(
        to_tinker_tool_specs(tools),
        system_prompt=system_prompt,
    )
    return list(prefix) + remaining


class TinkerRendererAdapter(BridgingRendererMixin):
    """Wrap a tinker-style renderer (tinker_cookbook / Fireworks cookbook)."""

    def __init__(self, inner: Any, *, close_token_ids: set[int] | None = None, synthesize_close: int | None = None):
        self._inner = inner
        stops = [int(t) for t in inner.get_stop_sequences()]
        self.close_token_ids = set(close_token_ids) if close_token_ids is not None else set(stops)
        self.synthesize_close = synthesize_close if synthesize_close is not None else (stops[0] if stops else None)

    def _with_tools(self, messages: list[dict], tools) -> list[dict]:
        return prepare_tinker_messages_with_tools(self._inner, messages, tools)

    def render_ids(self, messages, *, tools=None, add_generation_prompt: bool = False) -> list[int]:
        msgs = self._with_tools(to_tinker_messages(messages), tools)
        if add_generation_prompt:
            model_input = self._inner.build_generation_prompt(msgs)
        else:
            # Closed conversation (no trailing generation prompt); requires a
            # final assistant turn — satisfied by every internal caller.
            model_input = self._inner.build_supervised_example(msgs)[0]
        return list(model_input.to_ints())

    def render(self, messages, *, tools=None, add_generation_prompt: bool = False) -> RenderedTokens:
        return _render_with_prefix_attribution(
            self.render_ids,
            list(messages),
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

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
        return _render_with_prefix_attribution(
            self.render_ids,
            list(messages),
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

    def get_stop_token_ids(self) -> list[int]:
        return list(self.close_token_ids)

    def parse_response(self, token_ids: list[int], *, tools=None) -> ParsedResponse:
        return _to_parsed(self._tok.decode(list(token_ids), skip_special_tokens=True), None, None)
