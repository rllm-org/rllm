"""Unit tests for the rLLM parser-backed transport adapter."""

from types import SimpleNamespace

import pytest
from rllm_model_gateway.models import GatewayConfig
from rllm_model_gateway.proxy import _RllmParserTransport


class FakeParser:
    stop_sequences = [999]

    def __init__(self):
        self.parse_calls = []
        self.parsed_completion = {
            "content": "Hello from mock!",
            "reasoning": "",
            "tool_calls": [],
        }

    def parse(self, messages, **kwargs):
        self.parse_calls.append({"messages": messages, **kwargs})
        return "RAW_PROMPT"

    def parse_completion_text(self, text):
        self.last_completion_text = text
        return self.parsed_completion


def _transport(parser=None, **config_kwargs):
    config = GatewayConfig(
        model="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
        multi_turn_extension=True,
        accumulate_reasoning=True,
        **config_kwargs,
    )
    return _RllmParserTransport(config, parser=parser or FakeParser())


def _chat_body(**overrides):
    body = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "calc"}}],
        "temperature": 0.7,
        "max_completion_tokens": 16,
        "logprobs": True,
        "stop_token_ids": [123],
        "add_special_tokens": True,
    }
    body.update(overrides)
    return body


def _completion_response(**choice_overrides):
    choice = {
        "index": 0,
        "text": "Hello from mock!",
        "finish_reason": "stop",
        "token_ids": [10, 11, 12],
        "prompt_token_ids": [1, 2, 3],
        "logprobs": {
            "tokens": ["Hello", " from", " mock!"],
            "token_logprobs": [-0.5, -0.3, -0.1],
            "top_logprobs": [{"Hello": -0.5}, {" from": -0.3}, {" mock!": -0.1}],
        },
    }
    choice.update(choice_overrides)
    return {
        "id": "cmpl-test",
        "created": 123,
        "model": "mock-model",
        "choices": [choice],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3},
    }


def test_chat_to_completion_uses_raw_prompt_and_allowlist():
    parser = FakeParser()
    converted = _transport(parser).chat_to_completion(_chat_body(), originally_requested_logprobs=False)

    assert converted["prompt"] == "RAW_PROMPT"
    assert converted["max_tokens"] == 16
    assert converted["logprobs"] == 1
    assert converted["return_token_ids"] is True
    assert converted["add_special_tokens"] is False
    assert converted["stop_token_ids"] == [123, 999]
    assert "messages" not in converted
    assert parser.parse_calls[0]["tools"] == _chat_body()["tools"]


def test_chat_to_completion_preserves_top_logprobs_zero():
    converted = _transport().chat_to_completion(
        _chat_body(top_logprobs=0),
        originally_requested_logprobs=True,
    )
    assert converted["logprobs"] == 0


def test_chat_to_completion_always_requests_sampled_logprobs_for_trace():
    converted = _transport().chat_to_completion(
        _chat_body(logprobs=False),
        originally_requested_logprobs=False,
    )
    assert converted["logprobs"] == 1


def test_chat_to_completion_rejects_top_logprobs_without_original_logprobs():
    with pytest.raises(ValueError, match="top_logprobs requires logprobs=true"):
        _transport().chat_to_completion(
            _chat_body(top_logprobs=5),
            originally_requested_logprobs=False,
        )


def test_chat_to_completion_tool_choice_none_omits_tools():
    parser = FakeParser()
    _transport(parser).chat_to_completion(
        _chat_body(tool_choice="none"),
        originally_requested_logprobs=False,
    )
    assert parser.parse_calls[0]["tools"] == []


def test_chat_to_completion_rejects_required_tool_choice():
    with pytest.raises(ValueError, match="tool_choice"):
        _transport().chat_to_completion(
            _chat_body(tool_choice="required"),
            originally_requested_logprobs=False,
        )


def test_completion_to_chat_normalizes_logprobs_and_prompt_ids():
    parser = FakeParser()
    chat = _transport(parser).completion_to_chat(_completion_response())

    assert chat["object"] == "chat.completion"
    assert chat["prompt_token_ids"] == [1, 2, 3]
    choice = chat["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello from mock!"}
    assert choice["token_ids"] == [10, 11, 12]
    assert choice["logprobs"]["content"][0]["logprob"] == -0.5
    assert choice["logprobs"]["content"][0]["top_logprobs"] == [{"token": "Hello", "logprob": -0.5}]
    assert parser.last_completion_text == "Hello from mock!"


def test_completion_to_chat_converts_tool_calls_and_finish_reason():
    parser = FakeParser()
    parser.parsed_completion = {
        "content": "",
        "reasoning": "",
        "tool_calls": [SimpleNamespace(name="calculator", arguments={"expression": "1+1"})],
    }

    chat = _transport(parser).completion_to_chat(_completion_response())
    choice = chat["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_0_0",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'},
        }
    ]


@pytest.mark.parametrize(
    "logprobs",
    [
        None,
        {"tokens": ["Hello"], "token_logprobs": [-0.5]},
        {"tokens": ["Hello", " from", " mock!"], "token_logprobs": [-0.5, None, -0.1]},
    ],
)
def test_completion_to_chat_requires_dense_logprobs(logprobs):
    with pytest.raises(ValueError, match="logprobs|None|aligned"):
        _transport().completion_to_chat(_completion_response(logprobs=logprobs))


def test_completion_to_chat_preserves_truncated_mid_tool_as_text():
    parser = FakeParser()
    parser.parsed_completion = {
        "content": '<tool_call>{"name": "calculator"',
        "reasoning": "",
        "tool_calls": [],
    }

    chat = _transport(parser).completion_to_chat(_completion_response(finish_reason="length"))
    choice = chat["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["message"]["content"].startswith("<tool_call>")
    assert "tool_calls" not in choice["message"]
