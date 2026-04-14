"""
Tinker parser.

Wraps Tinker's Renderer to provide the same interface as ChatTemplateParser
(parse / parse_completion / stop_sequences).

Forward path (messages -> prompt string): converts OpenAI messages to Tinker
Messages via tinker-cookbook's openai_compat helpers, injects tool declarations
through the renderer when tools are provided, and renders via
Renderer.build_generation_prompt.

Reverse path (tokens -> structured response): Renderer.parse_response produces
a Tinker Message, Renderer.to_openai_message converts it to OpenAI-format
(each renderer subclass handles model-specific features like reasoning_content
and OpenAI-shaped tool_calls).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rllm.experimental.parser.utils import normalize_tools

logger = logging.getLogger(__name__)


class TinkerParser:
    def __init__(
        self,
        tokenizer,
        renderer_name: str | None = None,
        **kwargs,
    ):
        from tinker_cookbook import model_info, renderers

        self.tokenizer = tokenizer
        if renderer_name is None:
            renderer_name = model_info.get_recommended_renderer_name(tokenizer.name_or_path)
        self.renderer = renderers.get_renderer(renderer_name, tokenizer)
        self.stop_sequences: list[int] = self.renderer.get_stop_sequences()
        logger.info("TinkerParser: using renderer %r", renderer_name)

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, **kwargs) -> str:
        from tinker_cookbook.third_party.openai_compat import (
            openai_messages_to_tinker,
            openai_tools_to_tinker,
        )

        tools = kwargs.pop("tools", None)
        tools = normalize_tools(tools) if tools else []

        tinker_messages = openai_messages_to_tinker(messages)

        if tools:
            tool_specs = openai_tools_to_tinker(tools)
            system_prompt = ""
            if tinker_messages and tinker_messages[0]["role"] == "system":
                content = tinker_messages[0].get("content") or ""
                system_prompt = content if isinstance(content, str) else ""
                tinker_messages = tinker_messages[1:]
            prefix = self.renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt)
            tinker_messages = prefix + tinker_messages

        if add_generation_prompt:
            model_input = self.renderer.build_generation_prompt(tinker_messages)
        else:
            model_input, _weights = self.renderer.build_supervised_example(tinker_messages)

        tokens: list[int] = []
        for chunk in model_input.chunks:
            if hasattr(chunk, "tokens"):
                tokens.extend(chunk.tokens)
        # Tinker produces tokens natively; callers typically re-encode the
        # returned string back to token IDs. That round-trip is wasteful.
        # TODO: expose a way to return tokens directly to skip decode/encode.
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def parse_completion(self, completion_ids: list[int], **kwargs) -> dict[str, Any]:
        message, _success = self.renderer.parse_response(completion_ids)
        openai_msg = self.renderer.to_openai_message(message)

        return {
            "content": openai_msg.get("content") or "",
            "reasoning": openai_msg.get("reasoning_content") or "",
            "tool_calls": _convert_openai_tool_calls(openai_msg.get("tool_calls") or []),
        }


def _convert_openai_tool_calls(openai_tool_calls: list[dict]) -> list:
    """Convert OpenAI-format tool_call dicts to rllm ToolCall objects.

    Input shape (produced by Renderer.to_openai_message):
        {"type": "function", "id": ..., "function": {"name": ..., "arguments": str | dict}}
    """
    from rllm.tools.tool_base import ToolCall

    result = []
    for tc in openai_tool_calls:
        function = tc.get("function", {})
        name = function.get("name") or ""
        args_field = function.get("arguments", "")

        if isinstance(args_field, str):
            args_raw = args_field
            try:
                arguments = json.loads(args_field) if args_field else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        else:
            arguments = args_field or {}
            args_raw = json.dumps(arguments) if arguments else None

        result.append(ToolCall(name=name, arguments=arguments, arguments_raw=args_raw))
    return result
