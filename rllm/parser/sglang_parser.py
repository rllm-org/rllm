from __future__ import annotations

import json
import logging
from typing import Any

from rllm.parser.base import BaseChatParser, ChatMessage, ParsedCompletion, RenderedPrompt
from rllm.parser.utils import extract_images_pil, normalize_messages_for_images, normalize_tools
from rllm.tools.tool_base import ToolCall

logger = logging.getLogger(__name__)


class SGLangParser(BaseChatParser):
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
        self.reasoning_parser_name = reasoning_parser_name
        self.tool_parser_name = tool_parser_name
        self._skip_special_tokens = tool_parser_name is None

        if reasoning_parser_name is not None:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            if reasoning_parser_name.lower() not in ReasoningParser.DetectorMap:
                raise ValueError(f"Unknown SGLang reasoning parser: {reasoning_parser_name!r}")
            logger.info("SGLangParser: using reasoning parser %r", reasoning_parser_name)

        if tool_parser_name is not None:
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            if tool_parser_name not in FunctionCallParser.ToolCallParserEnum:
                raise ValueError(f"Unknown SGLang tool call parser: {tool_parser_name!r}")
            logger.info("SGLangParser: using tool parser %r (skip_special_tokens=%s)", tool_parser_name, self._skip_special_tokens)

    def render_messages(
        self,
        messages: list[ChatMessage],
        *,
        add_generation_prompt: bool = False,
        is_first_msg: bool = False,
        **kwargs,
    ) -> RenderedPrompt:
        tools = kwargs.pop("tools", None)
        if tools:
            tools = normalize_tools(tools)

        messages = normalize_messages_for_images(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return RenderedPrompt(token_ids=token_ids, text=prompt)

    def process_image_data(self, messages: list[dict]) -> list:
        if self.processor is None:
            raise RuntimeError("SGLangParser.process_image_data called without a multimodal processor")
        return extract_images_pil(normalize_messages_for_images(messages), self.processor)

    def parse_completion(self, completion_ids: list[int], **kwargs) -> ParsedCompletion:
        tools = kwargs.pop("tools", None)
        tools = normalize_tools(tools) if tools else []

        text = self.tokenizer.decode(completion_ids, skip_special_tokens=self._skip_special_tokens)
        reasoning: str | None = None
        content: str | None = text

        if self.reasoning_parser_name is not None:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser_name,
                stream_reasoning=False,
            )
            reasoning, content = reasoning_parser.parse_non_stream(text)

        tool_calls = []
        if self.tool_parser_name is not None and tools:
            from sglang.srt.entrypoints.openai.protocol import Tool
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            sglang_tools = [Tool.model_validate(tool) for tool in tools]
            function_call_parser = FunctionCallParser(
                tools=sglang_tools,
                tool_call_parser=self.tool_parser_name,
            )
            parse_input = content or ""
            if function_call_parser.has_tool_call(parse_input):
                content, tool_call_items = function_call_parser.parse_non_stream(parse_input)
                if tool_call_items:
                    tool_calls = _convert_sglang_tool_calls(tool_call_items)

        return ParsedCompletion(content=content or "", reasoning=reasoning or "", tool_calls=tool_calls)


def _convert_sglang_tool_calls(tool_call_items: list[Any]) -> list[ToolCall]:
    tool_calls = []
    for item in tool_call_items:
        try:
            arguments = json.loads(item.parameters)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        tool_calls.append(ToolCall(name=item.name or "", arguments=arguments))
    return tool_calls
