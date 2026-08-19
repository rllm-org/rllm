"""Tests for trace_converter: trace_record_to_step with tool_calls support."""

from rllm_model_gateway.models import TraceGraph, TraceRecord

from rllm.engine.trace_converter import (
    _parse_openai_tool_calls,
    filter_empty_response_traces,
    is_empty_response_trace,
    trace_delta_to_step_delta,
    trace_record_to_step,
)
from rllm.types import StepDelta, resolve_step_deltas

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

    def test_weight_version_propagated(self):
        trace = self._make_trace(weight_version=7)
        step = trace_record_to_step(trace)
        assert step.weight_version == 7
        assert step.model_output.weight_version == 7

    def test_weight_version_defaults_none(self):
        step = trace_record_to_step(self._make_trace())
        assert step.weight_version is None

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

    def test_step_with_reasoning_content_and_tool_call(self):
        """OpenAI-compatible GLM responses retain reasoning and tool calls."""
        trace = self._make_trace(
            response_message={
                "role": "assistant",
                "content": "I will inspect the repository.",
                "reasoning_content": "Let me think through this...",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "find . -maxdepth 2 -type d"}',
                        },
                    }
                ],
            },
            finish_reason="tool_calls",
        )
        step = trace_record_to_step(trace)

        assert step.thought == "Let me think through this..."
        assert step.model_output.reasoning == "Let me think through this..."
        assert step.model_output.tool_calls is not None
        assert step.model_output.tool_calls[0].name == "bash"
        assert step.model_output.tool_calls[0].arguments == {"command": "find . -maxdepth 2 -type d"}

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

    def test_trace_delta_maps_to_the_exact_flat_step(self):
        first = self._make_trace(trace_id="first", lineage_id="lineage")
        second = self._make_trace(
            trace_id="second",
            lineage_id="lineage",
            messages=[*first.messages, first.response_message, {"role": "user", "content": "weather?"}],
            prompt_token_ids=[*first.prompt_token_ids, *first.completion_token_ids, 4],
            response_message={
                "role": "assistant",
                "content": "",
                "reasoning_content": "Need the weather.",
                "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": '{"city":"London"}'}}],
            },
            completion_token_ids=[12],
            logprobs=[-0.2],
            finish_reason="tool_calls",
        )
        graph = TraceGraph(format="compact", version=1, deltas=[])
        graph.add(first)
        graph.add(second)

        actual = resolve_step_deltas([trace_delta_to_step_delta(delta) for delta in graph.deltas])
        expected = [trace_record_to_step(trace) for trace in (first, second)]

        assert [step.model_dump(mode="json") for step in actual] == [step.model_dump(mode="json") for step in expected]

    def test_trace_delta_preserves_missing_token_ids(self):
        trace = self._make_trace(prompt_token_ids=[], completion_token_ids=[], logprobs=None)
        graph = TraceGraph(format="compact", version=1, deltas=[])
        delta = trace_delta_to_step_delta(graph.add(trace))

        actual = resolve_step_deltas([StepDelta.model_validate_json(delta.model_dump_json())])[0]

        assert actual.model_dump(mode="json") == trace_record_to_step(trace).model_dump(mode="json")

    def test_arbitrary_metadata_does_not_override_graph_parentage(self):
        first = self._make_trace(trace_id="first", lineage_id=None, metadata={"lineage_id": "user-a"})
        second = self._make_trace(
            trace_id="second",
            lineage_id=None,
            metadata={"lineage_id": "user-b"},
            messages=[*first.messages, first.response_message],
            prompt_token_ids=[*first.prompt_token_ids, *first.completion_token_ids],
        )
        graph = TraceGraph(format="compact", version=1, deltas=[])
        graph.add(first)
        graph.add(second)

        actual = resolve_step_deltas([trace_delta_to_step_delta(delta) for delta in graph.deltas])

        assert graph.deltas[1].parent_trace_id == first.trace_id
        assert [step.model_dump(mode="json") for step in actual] == [trace_record_to_step(trace).model_dump(mode="json") for trace in (first, second)]


class TestEmptyResponseTraceFilter:
    def _make_trace(self, **overrides) -> TraceRecord:
        defaults = {
            "trace_id": "t-001",
            "session_id": "s-001",
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt_token_ids": [],
            "response_message": {},
            "completion_token_ids": [],
            "logprobs": [],
            "finish_reason": None,
        }
        defaults.update(overrides)
        return TraceRecord(**defaults)

    def test_detects_empty_response_without_consulting_logprobs(self):
        trace = self._make_trace(logprobs=[-0.1])

        assert is_empty_response_trace(trace)
        assert filter_empty_response_traces([trace]) == []

    def test_preserves_external_response_without_token_ids(self):
        trace = self._make_trace(
            response_message={"role": "assistant", "content": "answer"},
            finish_reason="stop",
        )

        assert not is_empty_response_trace(trace)
        assert filter_empty_response_traces([trace]) == [trace]

    def test_preserves_tool_only_response_with_empty_content(self):
        trace = self._make_trace(
            response_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                    }
                ],
            },
            finish_reason="tool_calls",
        )

        assert not is_empty_response_trace(trace)
