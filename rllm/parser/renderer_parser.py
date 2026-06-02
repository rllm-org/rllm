"""Token-native message parser backed by the external ``renderers`` package.

``renderers`` (https://www.primeintellect.ai/blog/renderers) converts messages
straight to token ids and — crucially for multi-turn RL — can extend a rollout
with ``bridge_to_next_turn`` so model-sampled tokens are reused verbatim
instead of being re-rendered through a chat template. Re-rendering silently
breaks token identity (boolean round-trips, BPE retokenization drift, dropped
reasoning blocks); the blog post lays out the failure modes.

:class:`RendererParser` adapts a ``renderers.Renderer`` to rLLM's
:class:`~rllm.parser.base.BaseParser` contract so it can stand in for
:class:`~rllm.parser.chat_template_parser.ChatTemplateParser`. It is
intentionally **not** yet wired into the rollout engine — this module just
puts the implementation in place behind the common contract.

Requires the ``renderers`` package: ``pip install 'renderers>=0.1.7'`` (also
pulled in by the ``verl`` / ``tinker`` optional dependency groups).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rllm.parser.base import BaseParser, Message, ParsedCompletion
from rllm.tools.tool_base import Tool, ToolCall, ToolOutput

if TYPE_CHECKING:
    from renderers import Renderer


class RendererParser(BaseParser):
    """Adapt a ``renderers.Renderer`` to the :class:`BaseParser` contract.

    Construct via :meth:`from_tokenizer` (you already hold a tokenizer) or
    :meth:`from_model` (let the renderers package load the tokenizer with its
    security + ``fastokens`` performance policy). The bare constructor is
    available when you want to supply a pre-built renderer directly.
    """

    def __init__(self, renderer: Renderer, tokenizer: Any = None):
        self.renderer = renderer
        self.tokenizer = tokenizer

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        *,
        renderer: str = "auto",
        **renderer_kwargs: Any,
    ) -> RendererParser:
        """Build a parser from an existing HuggingFace tokenizer.

        ``renderer`` selects the renderer by name (e.g. ``"qwen3"``,
        ``"gpt-oss"``) or ``"auto"`` to detect it from the tokenizer's model
        name. Extra kwargs (``preserve_all_thinking``, ``tool_parser``, ...)
        are forwarded to ``renderers.create_renderer``.
        """
        from renderers import create_renderer

        return cls(create_renderer(tokenizer, renderer=renderer, **renderer_kwargs), tokenizer=tokenizer)

    @classmethod
    def from_model(
        cls,
        model_name_or_path: str,
        *,
        renderer: str = "auto",
        **renderer_kwargs: Any,
    ) -> RendererParser:
        """Build a parser from a model name/path.

        The tokenizer is loaded via ``renderers.base.load_tokenizer``, which
        applies the package's ``trust_remote_code`` policy and the
        ``fastokens`` fast-encode patch.
        """
        from renderers import create_renderer
        from renderers.base import load_tokenizer

        tokenizer = load_tokenizer(model_name_or_path)
        return cls(create_renderer(tokenizer, renderer=renderer, **renderer_kwargs), tokenizer=tokenizer)

    # --- BaseParser contract ---------------------------------------------

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[Any] | None = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> list[int]:
        # The renderer handles reasoning accumulation via constructor flags
        # (preserve_all_thinking), so chat-template render kwargs such as
        # accumulate_reasoning / reasoning_effort are accepted and ignored.
        return list(
            self.renderer.render_ids(
                _to_renderer_messages(messages),
                tools=_to_renderer_tools(tools),
                add_generation_prompt=add_generation_prompt,
            )
        )

    def parse_completion(self, completion_ids: list[int]) -> ParsedCompletion:
        parsed = self.renderer.parse_response(list(completion_ids))
        return ParsedCompletion(
            content=parsed.content or "",
            reasoning=parsed.reasoning_content or None,
            tool_calls=_to_rllm_tool_calls(parsed.tool_calls),
        )

    def get_stop_token_ids(self) -> list[int]:
        return list(self.renderer.get_stop_token_ids())

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[Any] | None = None,
    ) -> list[int] | None:
        rendered = self.renderer.bridge_to_next_turn(
            list(previous_prompt_ids),
            list(previous_completion_ids),
            _to_renderer_messages(new_messages),
            tools=_to_renderer_tools(tools),
        )
        if rendered is None:
            return None
        return list(rendered.token_ids)

    @property
    def renderer_name(self) -> str:
        """Class name of the underlying renderer (e.g. ``"Qwen3Renderer"``)."""
        return type(self.renderer).__name__


# ---------------------------------------------------------------------------
# rLLM message/tool format <-> renderers format
#
# rLLM messages carry assistant reasoning under ``reasoning`` and tool results
# under ``tool_outputs``; tool calls may be ``ToolCall`` instances. The
# renderers package expects the OpenAI chat shape: ``reasoning_content``,
# string tool content, and ``{"type": "function", "function": {...}}`` tool
# calls. These helpers bridge the two and tolerate already-OpenAI-shaped input
# so the parser is a drop-in for either message style.
# ---------------------------------------------------------------------------


def _to_renderer_tools(tools: list[Any] | None) -> list[dict] | None:
    if not tools:
        return None
    out: list[dict] = []
    for tool in tools:
        if isinstance(tool, Tool):
            schema = tool.json
        elif isinstance(tool, str):
            schema = json.loads(tool)
        elif isinstance(tool, dict):
            schema = tool
        else:
            schema = getattr(tool, "json", None) or {}
        # renderers' ToolSpec is flat {name, description, parameters}; rLLM
        # tools follow OpenAI's {"type": "function", "function": {...}}.
        if isinstance(schema, dict) and "function" in schema:
            schema = schema["function"]
        out.append(schema)
    return out


def _to_renderer_messages(messages: list[Message]) -> list[dict]:
    return [_to_renderer_message(m) for m in messages]


def _to_renderer_message(message: Message) -> dict:
    role = message.get("role")
    out: dict[str, Any] = {"role": role}
    content = message.get("content")
    if role == "assistant":
        out["content"] = content or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            out["reasoning_content"] = reasoning
        tool_calls = message.get("tool_calls")
        if tool_calls:
            out["tool_calls"] = [_to_openai_tool_call(tc) for tc in tool_calls]
    elif role == "tool":
        out["content"] = content if content is not None else _stringify_tool_outputs(message.get("tool_outputs"))
        if message.get("name"):
            out["name"] = message["name"]
        if message.get("tool_call_id"):
            out["tool_call_id"] = message["tool_call_id"]
    else:
        out["content"] = content if content is not None else ""
    return out


def _to_openai_tool_call(tool_call: Any) -> dict:
    if isinstance(tool_call, ToolCall):
        function = {"name": tool_call.name, "arguments": tool_call.arguments}
    elif isinstance(tool_call, dict) and "function" in tool_call:
        return tool_call
    elif isinstance(tool_call, dict):
        function = {"name": tool_call.get("name", ""), "arguments": tool_call.get("arguments", {})}
    else:
        function = {"name": getattr(tool_call, "name", ""), "arguments": getattr(tool_call, "arguments", {})}
    return {"type": "function", "function": function}


def _to_rllm_tool_calls(tool_calls: Any) -> list[ToolCall]:
    if not tool_calls:
        return []
    out: list[ToolCall] = []
    for tc in tool_calls:
        if isinstance(tc, ToolCall):
            out.append(tc)
            continue
        if isinstance(tc, dict):
            function = tc.get("function", tc)
            name = function.get("name", "")
            arguments = function.get("arguments", {})
        else:
            name = getattr(tc, "name", "")
            arguments = getattr(tc, "arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                pass
        out.append(ToolCall(name=name, arguments=arguments))
    return out


def _stringify_tool_outputs(tool_outputs: Any) -> str:
    if not tool_outputs:
        return ""
    parts: list[str] = []
    for output in tool_outputs:
        if isinstance(output, ToolOutput):
            parts.append(str(output))
        elif isinstance(output, dict):
            parts.append(str(ToolOutput(**output)))
        else:
            parts.append(str(output))
    return "\n".join(parts)
