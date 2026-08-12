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


def _flatten_parts(content: Any) -> tuple[str, str]:
    """Split a renderer's content into ``(text, thinking)``.

    Parts concatenate without a separator — each carries its own whitespace.
    Ordering between text and thinking is lost, which the flat
    ``ParsedResponse`` shape cannot express; for the one-thinking-block-then-text
    turn every renderer here produces, the join is exact.
    """
    if not isinstance(content, list):
        return content or "", ""
    text = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    thinking = "".join(p.get("thinking", "") for p in content if isinstance(p, dict) and p.get("type") == "thinking")
    return text, thinking


def _to_parsed(content: Any, reasoning: str | None, tool_calls: Any) -> ParsedResponse:
    """Build a ``ParsedResponse``, flattening structured content.

    Tinker-style renderers return a turn's content as a parts list whenever the
    model emitted thinking, with the reasoning inside it rather than on
    ``reasoning_content``. ``ParsedResponse.content`` is a ``str`` and its
    consumers (``ModelOutput`` -> ``Step.model_response``) enforce that, so the
    parts are split here rather than left for every consumer to handle.
    """
    if isinstance(content, list):
        content, thinking = _flatten_parts(content)
        reasoning = reasoning or thinking
    return ParsedResponse(
        content=content or "",
        reasoning_content=reasoning or None,
        tool_calls=list(tool_calls) if tool_calls else None,
    )


def _sibling_reasoning(message: dict) -> str:
    """A turn's reasoning as harnesses echo it back, in wire-preference order."""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    provider = message.get("provider_specific_fields")
    value = provider.get("reasoning") if isinstance(provider, dict) else None
    return value if isinstance(value, str) else ""


def _is_qwen_history_renderer(renderer: Any) -> bool:
    """Whether ``renderer`` uses Qwen's last-real-user reasoning boundary."""
    return any(cls.__module__.startswith("tinker_cookbook.renderers.qwen3") for cls in type(renderer).__mro__)


def _text_content(content: Any) -> str | None:
    """Read wire text without mistaking multimodal content for a tool result."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    text: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            return None
        text.append(part["text"])
    return "".join(text)


def _is_tool_response_user(message: dict) -> bool:
    """Recognize the legacy Qwen wire shape where a tool result has user role."""
    content = _text_content(message.get("content"))
    return message.get("role") == "user" and content is not None and content.startswith("<tool_response>") and content.endswith("</tool_response>")


def _last_real_user_index(messages: list[dict]) -> int:
    """The boundary used by Qwen chat templates to retain recent thinking."""
    return max(
        (index for index, message in enumerate(messages) if message.get("role") == "user" and not _is_tool_response_user(message)),
        default=-1,
    )


def to_tinker_messages(
    messages: list[dict],
    *,
    lift_reasoning: bool = False,
    keep_reasoning_after: int | None = None,
) -> list[dict]:
    """Translate OpenAI wire messages into tinker-cookbook ``Message``s.

    Tinker-style renderers read a turn's tool calls only as pydantic
    ``ToolCall`` objects; harnesses send plain dicts, which raise on the first
    tool call in history.

    ``lift_reasoning`` additionally moves a turn's sibling ``reasoning_content``
    / ``reasoning`` into a structured ``thinking`` part, the only form a
    renderer reads it in. ``keep_reasoning_after`` applies Qwen's history rule:
    assistant thinking at or before that message index is removed, while later
    thinking is retained. It is off by default because lifting is only sound
    when the wrapped renderer keeps the supplied thinking parts.

    Messages already in cookbook form (parts-list content, ``ToolCall``
    objects) keep it; every other key rides along untouched, so cookbook-only
    fields (``trainable``, ``tools``, ``response_format``) survive the trip.
    """
    from tinker_cookbook.renderers.base import ToolCall

    modelled_tool_call_fields = set(ToolCall.model_fields)
    out: list[dict] = []
    for index, message in enumerate(messages):
        converted = {**message, "content": message.get("content") or ""}
        keep_reasoning = keep_reasoning_after is None or index > keep_reasoning_after
        content = converted["content"]
        if not keep_reasoning and message.get("role") == "assistant" and isinstance(content, list):
            converted["content"] = [part for part in content if not (isinstance(part, dict) and part.get("type") == "thinking")]
        reasoning = _sibling_reasoning(message) if lift_reasoning and keep_reasoning and message.get("role") == "assistant" else ""
        if reasoning:
            thinking = {"type": "thinking", "thinking": reasoning}
            content = converted["content"]
            if isinstance(content, str):
                converted["content"] = [thinking, {"type": "text", "text": content}]
            elif not any(isinstance(p, dict) and p.get("type") == "thinking" for p in content):
                converted["content"] = [thinking, *content]
        tool_calls = message.get("tool_calls")
        if tool_calls:
            # Wire tool calls carry fields the cookbook model forbids (e.g. the
            # streaming ``index``), so keep only what it declares.
            converted["tool_calls"] = [ToolCall.model_validate({k: v for k, v in tc.items() if k in modelled_tool_call_fields}) if isinstance(tc, dict) else tc for tc in tool_calls]
        out.append(converted)
    return out


def prepare_tinker_messages_for_history(
    renderer: Any,
    messages: list[dict],
    *,
    lift_reasoning: bool,
) -> list[dict]:
    """Apply the wrapped renderer's history policy while translating messages."""
    boundary = _last_real_user_index(messages) if _is_qwen_history_renderer(renderer) and not getattr(renderer, "strip_thinking_from_history", False) else None
    return to_tinker_messages(
        messages,
        lift_reasoning=lift_reasoning,
        keep_reasoning_after=boundary,
    )


def to_tinker_tool_specs(tools: list[dict] | None) -> list:
    """Translate OpenAI function declarations into cookbook ``ToolSpec``s."""
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


def prepare_tinker_messages_with_tools(
    renderer: Any,
    messages: list[dict],
    tools: list[dict] | None,
) -> list[dict]:
    """Inject declarations exactly as the cookbook serving renderer does."""
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
        # Cookbook Qwen strips thinking from every historical assistant turn,
        # while the HF template strips only at/before the last genuine user
        # query. Disable the coarse renderer behavior and apply the precise
        # boundary while translating messages. Tool-role observations therefore
        # preserve the current agent trace; a new user query clears earlier CoT.
        self._qwen_history_boundary = _is_qwen_history_renderer(inner)
        if self._qwen_history_boundary and getattr(inner, "strip_thinking_from_history", False):
            inner.strip_thinking_from_history = False
        self._lift_reasoning = not getattr(inner, "strip_thinking_from_history", False)
        stops = [int(t) for t in inner.get_stop_sequences()]
        self.close_token_ids = set(close_token_ids) if close_token_ids is not None else set(stops)
        self.synthesize_close = synthesize_close if synthesize_close is not None else (stops[0] if stops else None)

    def _with_tools(self, messages: list[dict], tools) -> list[dict]:
        return prepare_tinker_messages_with_tools(self._inner, messages, tools)

    def render_ids(self, messages, *, tools=None, add_generation_prompt: bool = False) -> list[int]:
        msgs = self._with_tools(
            prepare_tinker_messages_for_history(
                self._inner,
                messages,
                lift_reasoning=self._lift_reasoning,
            ),
            tools,
        )
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
        return _to_parsed(
            get("content", ""),
            get("reasoning_content"),
            get("tool_calls"),
        )


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
