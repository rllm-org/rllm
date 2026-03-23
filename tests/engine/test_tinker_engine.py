"""Tests for tinker_engine OpenAI-to-renderer conversion helpers."""

import json

import pytest
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import ToolCall as TinkerToolCall
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rllm.experimental.rollout.tinker_engine import (
    _convert_openai_messages,
    _parse_tinker_message,
    _prepare_messages_with_tools,
)
from rllm.tools.tool_base import ToolCall as RllmToolCall

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

CALCULATOR_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Compute math expressions",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

TOOL_CALL_OPENAI = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
}


def _make_tinker_tool_call(name: str = "calculator", arguments: str = '{"expression": "2+2"}', id: str = "call_0"):
    return TinkerToolCall(
        function=TinkerToolCall.FunctionBody(name=name, arguments=arguments),
        id=id,
    )


# ------------------------------------------------------------------
# _convert_openai_messages
# ------------------------------------------------------------------


class TestConvertOpenaiMessages:
    def test_tool_calls_become_pydantic_objects(self):
        """OpenAI tool_calls dicts should be converted to TinkerToolCall objects."""
        messages = [
            {"role": "assistant", "content": "Let me calculate.", "tool_calls": [TOOL_CALL_OPENAI]},
        ]
        result = _convert_openai_messages(messages)
        tc = result[0]["tool_calls"][0]
        assert isinstance(tc, TinkerToolCall)
        assert tc.function.name == "calculator"
        assert json.loads(tc.function.arguments) == {"expression": "2+2"}

    def test_tool_response_preserves_fields(self):
        """Tool response messages should preserve tool_call_id and name."""
        messages = [
            {"role": "tool", "content": "4", "tool_call_id": "call_0", "name": "calculator"},
        ]
        result = _convert_openai_messages(messages)
        assert result[0]["tool_call_id"] == "call_0"
        assert result[0]["name"] == "calculator"

    def test_array_content_passed_through(self):
        """Strands sends content as [{"type": "text", "text": "..."}] — list is truthy so it passes through."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
        ]
        result = _convert_openai_messages(messages)
        assert result[0]["content"] == [{"type": "text", "text": "What is 2+2?"}]

    def test_none_content_becomes_empty_string(self):
        messages = [{"role": "assistant", "content": None}]
        result = _convert_openai_messages(messages)
        assert result[0]["content"] == ""


# ------------------------------------------------------------------
# _prepare_messages_with_tools
# ------------------------------------------------------------------


class TestPrepareMessagesWithTools:
    @pytest.fixture()
    def qwen3_renderer(self):
        tok = get_tokenizer("Qwen/Qwen3-8B")
        return get_renderer("qwen3", tok, model_name="Qwen/Qwen3-8B")

    def test_tools_injected_into_system_message(self, qwen3_renderer):
        """Tool definitions should appear in the system message content."""
        messages = _convert_openai_messages(
            [
                {"role": "system", "content": "Solve math problems."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )
        result = _prepare_messages_with_tools(qwen3_renderer, messages, [CALCULATOR_TOOL_OPENAI])

        system_content = result[0]["content"]
        assert "calculator" in system_content
        assert "<tools>" in system_content
        assert "Solve math problems." in system_content

    def test_system_prompt_not_duplicated(self, qwen3_renderer):
        """Original system message should be replaced, not duplicated."""
        messages = _convert_openai_messages(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        result = _prepare_messages_with_tools(qwen3_renderer, messages, [CALCULATOR_TOOL_OPENAI])

        system_messages = [m for m in result if m["role"] == "system"]
        assert len(system_messages) == 1

    def test_no_system_message_creates_one(self, qwen3_renderer):
        """If no system message exists, one should be created with tool definitions."""
        messages = _convert_openai_messages(
            [
                {"role": "user", "content": "What is 2+2?"},
            ]
        )
        result = _prepare_messages_with_tools(qwen3_renderer, messages, [CALCULATOR_TOOL_OPENAI])

        assert result[0]["role"] == "system"
        assert "calculator" in result[0]["content"]
        assert result[-1]["role"] == "user"

    def test_non_function_tools_ignored(self, qwen3_renderer):
        """Tools without type='function' should be silently skipped."""
        messages = _convert_openai_messages(
            [
                {"role": "system", "content": "Hi"},
                {"role": "user", "content": "Hello"},
            ]
        )
        bad_tool = {"type": "retrieval", "name": "search"}
        result = _prepare_messages_with_tools(qwen3_renderer, messages, [bad_tool])

        # No tools injected, but system message still present
        assert "<tools>" not in result[0]["content"]


# ------------------------------------------------------------------
# _parse_tinker_message
# ------------------------------------------------------------------


class TestParseTinkerMessage:
    def test_tinker_tool_calls_converted_to_rllm(self):
        """Tinker ToolCall(function=FunctionBody(...)) should become rllm ToolCall(name, arguments)."""
        tc = _make_tinker_tool_call()
        message = {"role": "assistant", "content": "result", "tool_calls": [tc]}

        content, reasoning, tool_calls = _parse_tinker_message(message)
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], RllmToolCall)
        assert tool_calls[0].name == "calculator"
        assert tool_calls[0].arguments == {"expression": "2+2"}

    def test_structured_content_with_thinking(self):
        """List content with thinking and text parts should be separated."""
        message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me reason..."},
                {"type": "text", "text": "The answer is 4."},
            ],
        }
        content, reasoning, tool_calls = _parse_tinker_message(message)
        assert "answer is 4" in content
        assert "reason" in reasoning
        assert tool_calls == []

    def test_string_content_no_reasoning(self):
        """Plain string content should have empty reasoning."""
        message = {"role": "assistant", "content": "Hello world"}
        content, reasoning, tool_calls = _parse_tinker_message(message)
        assert content == "Hello world"
        assert reasoning == ""

    def test_multiple_tool_calls(self):
        """Multiple tool calls should all be converted."""
        tcs = [
            _make_tinker_tool_call("calculator", '{"expression": "2+2"}', "call_0"),
            _make_tinker_tool_call("calculator", '{"expression": "3*3"}', "call_1"),
        ]
        message = {"role": "assistant", "content": "Computing...", "tool_calls": tcs}
        _, _, tool_calls = _parse_tinker_message(message)
        assert len(tool_calls) == 2
        assert tool_calls[0].arguments == {"expression": "2+2"}
        assert tool_calls[1].arguments == {"expression": "3*3"}
