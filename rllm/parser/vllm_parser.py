from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from rllm.parser.base import BaseChatParser, ChatMessage, ParsedCompletion, RenderedPrompt
from rllm.parser.utils import extract_images_pil, normalize_messages_for_images, normalize_tools
from rllm.tools.tool_base import ToolCall

logger = logging.getLogger(__name__)


_VLLM_TOOL_PARSERS_NEEDING_SPECIAL_TOKENS = frozenset(
    {
        "deepseek_v32",
        "functiongemma",
        "glm45",
        "hermes",
        "internlm",
        "jamba",
        "mistral",
        "step3",
    }
)


@dataclass
class _VllmRequestStub:
    tools: list[dict] = field(default_factory=list)
    tool_choice: str = "auto"


class VLLMParser(BaseChatParser):
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
        self._reasoning_parser_cls = None
        self._tool_parser_cls = None
        self._skip_special_tokens = True

        if reasoning_parser_name is not None:
            from vllm.reasoning import ReasoningParserManager

            self._reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
            logger.info("VLLMParser: using reasoning parser %r", reasoning_parser_name)

        if tool_parser_name is not None:
            from vllm.tool_parsers import ToolParserManager

            self._tool_parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
            if tool_parser_name in _VLLM_TOOL_PARSERS_NEEDING_SPECIAL_TOKENS:
                self._skip_special_tokens = False
            logger.info("VLLMParser: using tool parser %r (skip_special_tokens=%s)", tool_parser_name, self._skip_special_tokens)

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
            raise RuntimeError("VLLMParser.process_image_data called without a multimodal processor")
        return extract_images_pil(normalize_messages_for_images(messages), self.processor)

    def parse_completion(self, completion_ids: list[int], **kwargs) -> ParsedCompletion:
        tools = kwargs.pop("tools", None)
        tools = normalize_tools(tools) if tools else []

        text = self.tokenizer.decode(completion_ids, skip_special_tokens=self._skip_special_tokens)
        reasoning: str | None = None
        content: str | None = text

        if self._reasoning_parser_cls is not None:
            reasoning_parser = self._reasoning_parser_cls(self.tokenizer)
            reasoning, content = reasoning_parser.extract_reasoning(text, request=None)

        tool_calls = []
        if self._tool_parser_cls is not None:
            tool_parser = self._tool_parser_cls(self.tokenizer)
            extracted = tool_parser.extract_tool_calls(content or "", request=_VllmRequestStub(tools=tools))
            if extracted.tools_called:
                tool_calls = _convert_vllm_tool_calls(extracted.tool_calls)
                content = extracted.content

        return ParsedCompletion(content=content or "", reasoning=reasoning or "", tool_calls=tool_calls)


def _convert_vllm_tool_calls(vllm_tool_calls: list[Any]) -> list[ToolCall]:
    tool_calls = []
    for tool_call in vllm_tool_calls:
        args_raw = tool_call.function.arguments
        try:
            arguments = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        tool_calls.append(ToolCall(name=tool_call.function.name, arguments=arguments))
    return tool_calls
