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

from typing import Any

from .bridging import BridgingRendererMixin
from .types import ParsedResponse, RenderedTokens


def _to_parsed(content: str, reasoning: str | None, tool_calls: Any) -> ParsedResponse:
    return ParsedResponse(
        content=content or "",
        reasoning_content=reasoning or None,
        tool_calls=list(tool_calls) if tool_calls else None,
    )


def _extract_tinker_message(message: Any) -> tuple[str, str, list[dict]]:
    """tinker_cookbook parsed message -> (content, reasoning, tool_calls dicts).

    Handles list content (text/thinking parts) and tinker ToolCall objects
    (function.name/arguments). Mirrors TinkerEngine's legacy _parse_tinker_message,
    emitting tool_calls as ``{"name","arguments"}`` dicts (accepted downstream)."""
    import json

    get = message.get if isinstance(message, dict) else (lambda k, d=None: getattr(message, k, d))
    content_field = get("content", "")
    if isinstance(content_field, list):
        texts = [p.get("text", "") for p in content_field if isinstance(p, dict) and p.get("type") == "text"]
        thinks = [p.get("thinking", "") for p in content_field if isinstance(p, dict) and p.get("type") == "thinking"]
        content, reasoning = "\n".join(texts), "\n".join(thinks)
    else:
        content, reasoning = content_field or "", ""

    tool_calls: list[dict] = []
    for tc in get("tool_calls", None) or []:
        if hasattr(tc, "function"):  # tinker_cookbook ToolCall(function=FunctionBody(...))
            args = tc.function.arguments
            tool_calls.append({"name": tc.function.name, "arguments": json.loads(args) if isinstance(args, str) else args})
        elif isinstance(tc, dict) and "function" in tc:  # OpenAI-shaped dict
            fn = tc["function"]
            args = fn.get("arguments", {})
            tool_calls.append({"name": fn.get("name", ""), "arguments": json.loads(args) if isinstance(args, str) else args})
        elif isinstance(tc, dict):  # already {"name","arguments"}
            tool_calls.append({"name": tc.get("name", ""), "arguments": tc.get("arguments", {})})
    return content, reasoning, tool_calls


class TinkerRendererAdapter(BridgingRendererMixin):
    """Wrap a tinker-style renderer (tinker_cookbook / Fireworks cookbook).

    Faithfully converts OpenAI-format messages and tools into the tinker format
    the inner renderer expects (tool_calls, tool results, images, tool specs),
    and converts parsed completions back — a drop-in for TinkerEngine's legacy
    render+parse path. (Prime-rl renderers take OpenAI messages natively and do
    not go through this adapter.)
    """

    def __init__(self, inner: Any, *, close_token_ids: set[int] | None = None, synthesize_close: int | None = None):
        self._inner = inner
        stops = [int(t) for t in inner.get_stop_sequences()]
        self.close_token_ids = set(close_token_ids) if close_token_ids is not None else set(stops)
        self.synthesize_close = synthesize_close if synthesize_close is not None else (stops[0] if stops else None)

    def _to_tinker_messages(self, messages: list[dict], tools) -> list[dict]:
        from tinker_cookbook.renderers.base import ToolCall as _TKToolCall
        from tinker_cookbook.renderers.base import ToolSpec as _TKToolSpec

        out: list[dict] = []
        for m in messages:
            content = m.get("content")
            if m.get("images"):  # rllm image format -> renderer content-part list
                content = [{"type": "image", "image": img} for img in m["images"]] + [{"type": "text", "text": content or ""}]
            tm: dict = {"role": m["role"], "content": content if content is not None else ""}
            if "name" in m:
                tm["name"] = m["name"]
            if "tool_call_id" in m:
                tm["tool_call_id"] = m["tool_call_id"]
            if m.get("tool_calls"):
                tm["tool_calls"] = [_TKToolCall.model_validate(tc) for tc in m["tool_calls"]]
            out.append(tm)
        if tools:
            specs = [
                _TKToolSpec(name=t["function"]["name"], description=t["function"].get("description", ""), parameters=t["function"].get("parameters", {})) for t in tools if t.get("type") == "function"
            ]
            system_prompt = ""
            if out and out[0]["role"] == "system":
                sp = out[0].get("content") or ""
                system_prompt = sp if isinstance(sp, str) else ""
                out = out[1:]
            out = list(self._inner.create_conversation_prefix_with_tools(specs, system_prompt)) + out
        return out

    def render_ids(self, messages, *, tools=None, add_generation_prompt: bool = False) -> list[int]:
        msgs = self._to_tinker_messages(list(messages), tools)
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
        return _to_parsed(*_extract_tinker_message(msg))


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
