"""
SGLang parser.

Wraps SGLang's ReasoningParser and FunctionCallParser to provide
the same interface as ChatTemplateParser (parse / parse_completion).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rllm.experimental.parser.utils import extract_images_pil, normalize_messages_for_images, normalize_tools

logger = logging.getLogger(__name__)


class SGLangParser:
    def __init__(
        self,
        tokenizer,
        reasoning_parser_name: str | None = None,
        tool_parser_name: str | None = None,
        processor=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self._reasoning_parser_name = reasoning_parser_name
        self._tool_parser_name = tool_parser_name
        # SGLang flips skip_special_tokens=False unconditionally whenever tools
        # are present in a request. We mirror that with a single rule: if a
        # tool parser is configured, decode with special tokens preserved.
        self._skip_special_tokens = tool_parser_name is None

        if reasoning_parser_name is not None:
            # Import eagerly to fail-fast if the name is invalid.
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            if reasoning_parser_name.lower() not in ReasoningParser.DetectorMap:
                raise ValueError(f"Unknown SGLang reasoning parser: {reasoning_parser_name!r}")
            logger.info("SGLangParser: using reasoning parser %r", reasoning_parser_name)

        if tool_parser_name is not None:
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            if tool_parser_name not in FunctionCallParser.ToolCallParserEnum:
                raise ValueError(f"Unknown SGLang tool call parser: {tool_parser_name!r}")
            logger.info(
                "SGLangParser: using tool parser %r (skip_special_tokens=%s)",
                tool_parser_name,
                self._skip_special_tokens,
            )

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, **kwargs) -> str:
        tools = kwargs.pop("tools", None)
        if tools:
            tools = normalize_tools(tools)

        messages = normalize_messages_for_images(messages)

        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def process_image_data(self, messages: list[dict]) -> list:
        """Resolve image payloads in messages to PIL images. Requires a processor."""
        if self.processor is None:
            raise RuntimeError("SGLangParser.process_image_data called without a multimodal processor")
        messages = normalize_messages_for_images(messages)
        return extract_images_pil(messages, self.processor)

    def parse_completion(self, completion_ids: list[int], **kwargs) -> dict[str, Any]:
        tools = kwargs.pop("tools", None)
        tools = normalize_tools(tools) if tools else []

        text = self.tokenizer.decode(completion_ids, skip_special_tokens=self._skip_special_tokens)

        reasoning: str | None = None
        content: str | None = text

        if self._reasoning_parser_name is not None:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            reasoning_parser = ReasoningParser(
                model_type=self._reasoning_parser_name,
                stream_reasoning=False,
            )
            reasoning, content = reasoning_parser.parse_non_stream(text)

        tool_calls: list | None = None
        if self._tool_parser_name is not None and tools:
            from sglang.srt.entrypoints.openai.protocol import Tool
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            sglang_tools = [Tool.model_validate(t) for t in tools]
            function_call_parser = FunctionCallParser(
                tools=sglang_tools,
                tool_call_parser=self._tool_parser_name,
            )
            parse_input = content if content is not None else ""
            if function_call_parser.has_tool_call(parse_input):
                content, tool_call_items = function_call_parser.parse_non_stream(parse_input)
                if tool_call_items:
                    tool_calls = _convert_sglang_tool_calls(tool_call_items)

        return {
            "content": content or "",
            "reasoning": reasoning or "",
            "tool_calls": tool_calls or [],
        }


def _convert_sglang_tool_calls(tool_call_items: list) -> list:
    """Convert SGLang ToolCallItem objects to rllm ToolCall objects.

    SGLang ToolCallItem has: name (str | None), parameters (str, JSON).
    """
    from rllm.tools.tool_base import ToolCall

    result = []
    for item in tool_call_items:
        try:
            arguments = json.loads(item.parameters)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        result.append(ToolCall(name=item.name or "", arguments=arguments, arguments_raw=item.parameters))
    return result
