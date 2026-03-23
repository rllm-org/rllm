"""Tests for trace_converter: trace_record_to_step with tool_calls support."""

from rllm_model_gateway.models import TraceRecord

from rllm.experimental.engine.trace_converter import (
    _parse_openai_tool_calls,
    trace_record_to_step,
)

# ------------------------------------------------------------------
# _parse_openai_tool_calls
# ------------------------------------------------------------------


class TestParseOpenaiToolCalls:
    def test_basic_conversion(self):
        raw = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
            }
        ]
        result = _parse_openai_tool_calls(raw)
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"city": "London"}

    def test_multiple_tool_calls(self):
        raw = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "test"}'},
            },
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calc", "arguments": '{"expr": "1+1"}'},
            },
        ]
        result = _parse_openai_tool_calls(raw)
        assert len(result) == 2
        assert result[0].name == "search"
        assert result[1].name == "calc"
        assert result[1].arguments == {"expr": "1+1"}

    def test_invalid_json_arguments(self):
        raw = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "foo", "arguments": "not-json"},
            }
        ]
        result = _parse_openai_tool_calls(raw)
        assert result[0].name == "foo"
        assert result[0].arguments == {"raw": "not-json"}

    def test_dict_arguments(self):
        """Arguments already parsed as dict (e.g. from in-process handler)."""
        raw = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "bar", "arguments": {"x": 1}},
            }
        ]
        result = _parse_openai_tool_calls(raw)
        assert result[0].arguments == {"x": 1}

    def test_empty_list(self):
        assert _parse_openai_tool_calls([]) == []


# ------------------------------------------------------------------
# trace_record_to_step with tool_calls
# ------------------------------------------------------------------


class TestTraceRecordToStep:
    def _make_trace(self, **overrides) -> TraceRecord:
        defaults = {
            "trace_id": "t-001",
            "session_id": "s-001",
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt_token_ids": [1, 2, 3],
            "response_message": {
                "role": "assistant",
                "content": "Hi there!",
            },
            "completion_token_ids": [10, 11],
            "logprobs": [-0.5, -0.3],
            "finish_reason": "stop",
        }
        defaults.update(overrides)
        return TraceRecord(**defaults)

    def test_basic_step(self):
        trace = self._make_trace()
        step = trace_record_to_step(trace)

        assert step.id == "t-001"
        assert step.model_response == "Hi there!"
        assert step.model_output.content == "Hi there!"
        assert step.model_output.prompt_ids == [1, 2, 3]
        assert step.model_output.completion_ids == [10, 11]
        assert step.model_output.logprobs == [-0.5, -0.3]
        assert step.model_output.tool_calls is None

    def test_step_with_tool_calls(self):
        trace = self._make_trace(
            response_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    },
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expr": "2+2"}',
                        },
                    },
                ],
            },
            finish_reason="tool_calls",
        )
        step = trace_record_to_step(trace)

        assert step.model_output.tool_calls is not None
        assert len(step.model_output.tool_calls) == 2
        assert step.model_output.tool_calls[0].name == "get_weather"
        assert step.model_output.tool_calls[0].arguments == {"city": "London"}
        assert step.model_output.tool_calls[1].name == "calculate"
        assert step.model_output.tool_calls[1].arguments == {"expr": "2+2"}
        assert step.model_output.finish_reason == "tool_calls"

    def test_step_with_reasoning(self):
        trace = self._make_trace(
            response_message={
                "role": "assistant",
                "content": "42",
                "reasoning": "Let me think...",
            },
        )
        step = trace_record_to_step(trace)
        assert step.thought == "Let me think..."
        assert step.model_output.reasoning == "Let me think..."

    def test_chat_completions_includes_response(self):
        trace = self._make_trace()
        step = trace_record_to_step(trace)
        assert len(step.chat_completions) == 2  # user msg + assistant msg
        assert step.chat_completions[-1]["role"] == "assistant"

    def test_no_tool_calls_key_means_none(self):
        """If response_message has no tool_calls key, model_output.tool_calls should be None."""
        trace = self._make_trace(
            response_message={"role": "assistant", "content": "just text"},
        )
        step = trace_record_to_step(trace)
        assert step.model_output.tool_calls is None
