import json
import uuid
from collections.abc import Sequence
from typing import Any

from renderers import ParsedToolCall, config_from_name, create_renderer
from transformers import AutoTokenizer

from rllm_model_gateway.v2.config import TokenizationConfig


class TokenizationService:
    def __init__(self, config: TokenizationConfig) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
        )
        self._renderer = create_renderer(
            self._tokenizer,
            config_from_name(config.renderer),
            chat_template_kwargs=config.renderer_kwargs or None,
        )

    def encode(self, prompt: str) -> list[int]:
        return list(self._tokenizer.encode(prompt, add_special_tokens=False))

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[int]:
        return list(
            self._renderer.render_ids(
                messages,
                tools=_to_renderer_tools(tools),
                add_generation_prompt=True,
            )
        )

    def bridge(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[int] | None:
        rendered = self._renderer.bridge_to_next_turn(
            previous_prompt_ids,
            previous_completion_ids,
            new_messages,
            tools=_to_renderer_tools(tools),
        )
        if rendered is None:
            return None
        return list(rendered.token_ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def parse_completion(self, token_ids: list[int], tools: list[dict[str, Any]]) -> dict[str, Any]:
        parsed = self._renderer.parse_response(token_ids, tools=_to_renderer_tools(tools))
        return {
            "content": parsed.content,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": _to_api_tool_calls(parsed.tool_calls),
        }

    def stop_token_ids(self) -> list[int]:
        return list(self._renderer.get_stop_token_ids())


def _to_renderer_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        nested = tool.get("function")
        function = nested if tool.get("type") == "function" and isinstance(nested, dict) else tool
        if not isinstance(function, dict) or not function.get("name"):
            continue
        converted.append(
            {
                "name": str(function["name"]),
                "description": str(function.get("description", "")),
                "parameters": dict(function.get("parameters") or {}),
            }
        )
    return converted or None


def _to_api_tool_calls(parsed_tool_calls: list[ParsedToolCall]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for call in parsed_tool_calls:
        if not call.name:
            continue
        arguments = call.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, separators=(",", ":"))
        tool_calls.append(
            {
                "id": call.id or f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": call.name, "arguments": arguments},
            }
        )
    return tool_calls
