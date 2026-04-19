"""
vLLM parser.

Wraps vLLM's ReasoningParserManager and ToolParserManager to provide
the same interface as ChatTemplateParser (parse / parse_completion).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from rllm.experimental.parser.utils import extract_images_pil, normalize_messages_for_images, normalize_tools

logger = logging.getLogger(__name__)

# vLLM tool parsers whose adjust_request() sets skip_special_tokens=False
# because their tool-call format relies on special tokens being present
# in the decoded text. Keep in sync with vLLM upstream.
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
    """Minimal stand-in for vLLM's ChatCompletionRequest.

    Several tool parsers read request.tools inside extract_tool_calls
    (seed_oss, internlm2, qwen3xml, qwen3coder, glm4_moe, step3p5).
    """

    tools: list[dict] = field(default_factory=list)
    tool_choice: str = "auto"


class VLLMParser:
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
            logger.info(
                "VLLMParser: using tool parser %r (skip_special_tokens=%s)",
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
            raise RuntimeError("VLLMParser.process_image_data called without a multimodal processor")
        messages = normalize_messages_for_images(messages)
        return extract_images_pil(messages, self.processor)

    def parse_completion(self, completion_ids: list[int], **kwargs) -> dict[str, Any]:
        tools = kwargs.pop("tools", None)
        tools = normalize_tools(tools) if tools else []

        text = self.tokenizer.decode(completion_ids, skip_special_tokens=self._skip_special_tokens)

        reasoning: str | None = None
        content: str | None = text

        if self._reasoning_parser_cls is not None:
            reasoning_parser = self._reasoning_parser_cls(self.tokenizer)
            reasoning, content = reasoning_parser.extract_reasoning(text, request=None)

        tool_calls: list | None = None
        if self._tool_parser_cls is not None:
            tool_parser = self._tool_parser_cls(self.tokenizer)
            request_stub = _VllmRequestStub(tools=tools)
            extracted = tool_parser.extract_tool_calls(content if content is not None else "", request=request_stub)
            if extracted.tools_called:
                tool_calls = _convert_vllm_tool_calls(extracted.tool_calls)
                content = extracted.content

        return {
            "content": content or "",
            "reasoning": reasoning or "",
            "tool_calls": tool_calls or [],
        }


def _convert_vllm_tool_calls(vllm_tool_calls: list) -> list:
    from rllm.tools.tool_base import ToolCall

    result = []
    for tc in vllm_tool_calls:
        args_raw = tc.function.arguments
        try:
            arguments = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        result.append(ToolCall(name=tc.function.name, arguments=arguments, arguments_raw=args_raw))
    return result
