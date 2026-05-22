from __future__ import annotations

import json
from typing import Any

from rllm.parser.base import BaseChatParser, ChatMessage, ParsedCompletion, RenderedPrompt
from rllm.parser.utils import normalize_tools
from rllm.tools.tool_base import ToolCall


class TinkerParser(BaseChatParser):
    """Wrap a tinker-cookbook renderer with the chat parser interface."""

    def __init__(
        self,
        tokenizer,
        renderer_name: str | None = None,
        image_processor=None,
        **renderer_kwargs,
    ):
        from tinker_cookbook import model_info, renderers

        self.tokenizer = tokenizer
        if renderer_name is None:
            renderer_name = model_info.get_recommended_renderer_name(tokenizer.name_or_path)

        self.renderer_name = renderer_name
        self.renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor=image_processor, **renderer_kwargs)
        self.stop_sequences: list[int] = self.renderer.get_stop_sequences()

    def render_messages(
        self,
        messages: list[ChatMessage],
        *,
        add_generation_prompt: bool = False,
        is_first_msg: bool = False,
        **kwargs,
    ) -> RenderedPrompt:
        tools = normalize_tools(kwargs.pop("tools", None) or [])
        model_input = self._build_model_input(messages, add_generation_prompt=add_generation_prompt, tools=tools)

        try:
            token_ids = model_input.to_ints()
        except ValueError as err:
            raise ValueError("TinkerParser.render_messages cannot represent non-text ModelInput chunks as RenderedPrompt token_ids") from err

        return RenderedPrompt(
            token_ids=token_ids,
            text=self.tokenizer.decode(token_ids, skip_special_tokens=False),
            metadata={"renderer_name": self.renderer_name},
        )

    def parse_completion(self, completion_ids: list[int], **kwargs) -> ParsedCompletion:
        message, _success = self.renderer.parse_response(completion_ids)

        to_openai_message = getattr(self.renderer, "to_openai_message", None)
        if callable(to_openai_message):
            openai_message = to_openai_message(message)
            return ParsedCompletion(
                content=openai_message.get("content") or "",
                reasoning=openai_message.get("reasoning_content") or openai_message.get("reasoning") or "",
                tool_calls=_convert_openai_tool_calls(openai_message.get("tool_calls") or []),
            )

        content, reasoning, tool_calls = _parse_tinker_message(message)
        return ParsedCompletion(content=content, reasoning=reasoning, tool_calls=tool_calls)

    def _build_model_input(self, messages: list[ChatMessage], *, add_generation_prompt: bool, tools: list[dict]):
        tinker_messages = _openai_messages_to_tinker(_normalize_messages_for_tinker(messages))
        if tools:
            tinker_messages = _prepare_messages_with_tools(self.renderer, tinker_messages, tools)

        if add_generation_prompt:
            return self.renderer.build_generation_prompt(tinker_messages)

        model_input, _weights = self.renderer.build_supervised_example(tinker_messages)
        return model_input


def _normalize_messages_for_tinker(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        content = message.get("content")
        images = message.get("images")
        if not images:
            normalized.append(dict(message))
            continue

        parts = []
        if isinstance(content, str) and content:
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            parts.extend(content)

        for image in images:
            if isinstance(image, dict) and "image" in image:
                parts.append({"type": "image", "image": image["image"]})
            else:
                parts.append({"type": "image", "image": image})

        normalized_message = dict(message)
        normalized_message["content"] = parts
        normalized_message.pop("images", None)
        normalized.append(normalized_message)
    return normalized


def _openai_messages_to_tinker(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from tinker_cookbook.third_party.openai_compat import openai_messages_to_tinker

        return openai_messages_to_tinker(messages)
    except ImportError:
        return [_fallback_openai_message_to_tinker(message) for message in messages]


def _fallback_openai_message_to_tinker(message: dict[str, Any]) -> dict[str, Any]:
    from tinker_cookbook.renderers.base import ToolCall as TinkerToolCall

    tinker_message = {
        "role": message["role"],
        "content": message.get("content") or "",
    }
    if "name" in message:
        tinker_message["name"] = message["name"]
    if "tool_call_id" in message:
        tinker_message["tool_call_id"] = message["tool_call_id"]
    if "tool_calls" in message:
        tinker_message["tool_calls"] = [TinkerToolCall.model_validate(tool_call) for tool_call in message["tool_calls"]]
    return tinker_message


def _prepare_messages_with_tools(renderer, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker

        tool_specs = openai_tools_to_tinker(tools)
    except ImportError:
        from tinker_cookbook.renderers.base import ToolSpec

        tool_specs = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool["function"]
            tool_specs.append(ToolSpec(name=function["name"], description=function.get("description", ""), parameters=function.get("parameters", {})))

    system_prompt = ""
    if messages and messages[0]["role"] == "system":
        content = messages[0].get("content") or ""
        system_prompt = content if isinstance(content, str) else ""
        messages = list(messages[1:])

    return renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt) + messages


def _parse_tinker_message(message: dict[str, Any]) -> tuple[str, str, list[ToolCall]]:
    content = message["content"]
    if isinstance(content, list):
        text_parts = [part["text"] for part in content if part["type"] == "text"]
        thinking_parts = [part["thinking"] for part in content if part["type"] == "thinking"]
        parsed_content = "\n".join(text_parts)
        reasoning = "\n".join(thinking_parts)
    else:
        parsed_content = content
        reasoning = ""

    return parsed_content, reasoning, _convert_tinker_tool_calls(message.get("tool_calls") or [])


def _convert_tinker_tool_calls(tool_calls: list[Any]) -> list[ToolCall]:
    converted = []
    for tool_call in tool_calls:
        if isinstance(tool_call, ToolCall):
            converted.append(tool_call)
        elif hasattr(tool_call, "function"):
            args = tool_call.function.arguments
            converted.append(ToolCall(name=tool_call.function.name, arguments=json.loads(args) if isinstance(args, str) else args))
        elif isinstance(tool_call, dict):
            converted.append(ToolCall(name=tool_call.get("name", ""), arguments=tool_call.get("arguments", {})))
        else:
            raise TypeError(f"Unrecognized tool_call type: {type(tool_call)}")
    return converted


def _convert_openai_tool_calls(tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
    converted = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                arguments = {}
        converted.append(ToolCall(name=function.get("name", ""), arguments=arguments))
    return converted
