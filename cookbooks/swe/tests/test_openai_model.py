#!/usr/bin/env python3
"""Tests for the OpenAI-compatible mini-swe-agent model wrapper."""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import pytest
from minisweagent.exceptions import LimitsExceeded
from rllm.workflows.workflow import TerminationReason

from swe.environment import ensure_bootstrapped

ensure_bootstrapped()

from swe.agent_flow import ProgressLoggingAgent
from swe.flow_config import add_flow_cli_args, flow_config_from_args
from swe.openai_model import MaxPromptLengthExceeded, MaxResponseLengthExceeded, OpenAIClientModel
from swe.utils import (
    build_error_details,
    classify_termination,
)


class FakeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, **kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


class FakeMessage:
    def __init__(self, *, content=None, tool_calls=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning = reasoning

    def model_dump(self, **kwargs):
        data = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [dump_tool_call(tool_call) for tool_call in self.tool_calls or []],
            "refusal": None,
            "reasoning": self.reasoning,
        }
        if kwargs.get("exclude_none"):
            return {key: value for key, value in data.items() if value is not None}
        return data


class FakeResponse:
    def __init__(self, *, message, finish_reason="tool_calls", token_ids=None):
        self.choices = [SimpleNamespace(message=message, finish_reason=finish_reason)]
        self._token_ids = token_ids

    def model_dump(self):
        choice = {
            "message": self.choices[0].message.model_dump(),
            "finish_reason": self.choices[0].finish_reason,
        }
        if self._token_ids is not None:
            choice["token_ids"] = list(self._token_ids)
        return {"choices": [choice], "usage": {"prompt_tokens": 7}}


class FakeCompletions:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, *responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(*responses))


class FailingCompletions:
    def __init__(self, message):
        self.message = message
        self.calls = []

    def create(self, **kwargs):
        from openai import BadRequestError

        self.calls.append(kwargs)
        raise BadRequestError(
            message=self.message,
            response=SimpleNamespace(request=None, status_code=400, headers={}),
            body={"error": {"message": self.message}},
        )


class DummyEnv:
    def execute(self, action, **kwargs):
        return {"returncode": 0, "output": "", "exception_info": None}

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class MaxResponseModel:
    cost = 0.0
    n_calls = 0

    def query(self, messages):
        raise MaxResponseLengthExceeded()

    def format_message(self, **kwargs):
        return dict(kwargs)

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class MaxPromptModel(MaxResponseModel):
    def query(self, messages):
        raise MaxPromptLengthExceeded()


def tool_call(arguments='{"command": "pwd"}', name="bash"):
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def dump_tool_call(call):
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": call.function.arguments,
        },
    }


def make_model(
    response,
    *,
    model_config=None,
    base_url="http://example.test/v1",
    model_name="Qwen/Qwen3.5-9B",
):
    model = OpenAIClientModel(
        base_url=base_url,
        model_name=model_name,
        api_key="test",
        model_config=model_config or {},
    )
    model.client = FakeClient(response)
    return model


def test_api_path_does_not_load_tokenizer(monkeypatch):
    response = FakeResponse(message=FakeMessage(content=None, tool_calls=[tool_call()]))
    model = make_model(response)
    monkeypatch.setattr(model, "_get_tokenizer", lambda: pytest.fail("tokenizer loaded"))

    message = model.query([{"role": "user", "content": "hi"}])

    assert message["extra"]["raw_transcript"] is False
    assert message["extra"]["actions"] == [{"command": "pwd", "tool_call_id": "call_1"}]


def test_query_sanitizes_messages_and_request_kwargs():
    response = FakeResponse(message=FakeMessage(content=None, tool_calls=[tool_call()]))
    model = make_model(
        response,
        model_config={
            "model_kwargs": {
                "temperature": 0.2,
                "max_tokens": 17,
                "n": 2,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
    )

    model.query([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [dump_tool_call(tool_call())],
            "extra": {"local": True},
            "refusal": None,
            "reasoning": "local-only",
        }
    ])

    call = model.client.chat.completions.calls[0]
    assert call["messages"] == [{"role": "assistant", "tool_calls": [dump_tool_call(tool_call())]}]
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 17
    assert "n" not in call
    assert call["parallel_tool_calls"] is False
    assert call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "return_token_ids" not in call["extra_body"]


def test_official_openai_api_does_not_receive_extra_body():
    response = FakeResponse(message=FakeMessage(content=None, tool_calls=[tool_call()]))
    model = make_model(
        response,
        base_url="https://api.openai.com/v1",
        model_name="gpt-5-mini",
        model_config={
            "model_kwargs": {
                "temperature": 0.0,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
    )

    model.query([{"role": "user", "content": "hi"}])

    call = model.client.chat.completions.calls[0]
    assert "extra_body" not in call
    assert "chat_template_kwargs" not in call
    assert "temperature" not in call


def test_return_token_ids_is_opt_in_for_compatible_gateways():
    response = FakeResponse(message=FakeMessage(content=None, tool_calls=[tool_call()]))
    model = make_model(
        response,
        model_config={"model_kwargs": {"return_token_ids": True}},
    )

    model.query([{"role": "user", "content": "hi"}])

    call = model.client.chat.completions.calls[0]
    assert call["extra_body"]["return_token_ids"] is True


def test_request_parameter_errors_are_not_format_errors():
    assert OpenAIClientModel._is_request_parameter_error("Unknown parameter: chat_template_kwargs")
    assert OpenAIClientModel._is_request_parameter_error("Unsupported value: temperature")


def test_query_context_length_bad_request_raises_max_prompt_length():
    model = OpenAIClientModel(
        base_url="https://api.openai.com/v1",
        model_name="gpt-5-mini",
        api_key="test",
    )
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FailingCompletions("maximum context length exceeded")
        )
    )

    with pytest.raises(MaxPromptLengthExceeded):
        model.query([{"role": "user", "content": "hi"}])


def test_native_finish_reason_length_raises():
    response = FakeResponse(message=FakeMessage(content="partial"), finish_reason="length")
    model = make_model(response)

    with pytest.raises(MaxResponseLengthExceeded):
        model.query([{"role": "user", "content": "hi"}])


def test_token_aware_length_finish_reason_raises_max_response_length():
    response = FakeResponse(
        message=FakeMessage(content="", tool_calls=[]),
        finish_reason="length",
        token_ids=[1, 2, 3],
    )
    model = make_model(response)
    model.tokenizer = FakeTokenizer()

    with pytest.raises(MaxResponseLengthExceeded):
        model.query([{"role": "user", "content": "hi"}])


def test_token_aware_eos_decodes_raw_transcript_and_strips_tool_calls():
    response = FakeResponse(
        message=FakeMessage(content="", tool_calls=[tool_call()]),
        token_ids=[10, 11, FakeTokenizer.eos_token_id],
    )
    model = make_model(response)
    model.tokenizer = FakeTokenizer()

    message = model.query([{"role": "user", "content": "hi"}])

    assert message["content"] == "10 11"
    assert "tool_calls" not in message
    assert message["extra"]["raw_transcript"] is True
    assert message["extra"]["completion_token_ids"] == [10, 11, 99]


def test_format_error_content_is_non_null_and_strips_tool_calls():
    response = FakeResponse(
        message=FakeMessage(content=None, tool_calls=[tool_call(arguments="{bad json")]),
    )
    model = make_model(response)

    message = model.query([{"role": "user", "content": "hi"}])

    assert message["content"] == ""
    assert "tool_calls" not in message
    assert message["extra"]["actions"] == []
    assert "format_error" in message["extra"]


def test_summarize_context_detects_length_finish_reason():
    response = FakeResponse(message=FakeMessage(content="partial"), finish_reason="length")
    model = make_model(response)

    with pytest.raises(MaxResponseLengthExceeded):
        model.summarize_context([{"role": "user", "content": "old"}], "summarize")


def test_summarize_context_uses_standard_tool_request_and_strips_thinking():
    text = "<think>\nchoose durable facts\n</think>\nKeep src/a.py fix."
    response = FakeResponse(message=FakeMessage(content=text), finish_reason="stop")
    model = make_model(
        response,
        model_config={
            "compaction_continuation_template": (
                "You are continuing from a compacted earlier session.\n\n"
                "<compact_context>\n"
                "{{ summary }}\n"
                "</compact_context>"
            )
        },
    )

    message = model.summarize_context([{"role": "user", "content": "old"}], "summarize")
    call = model.client.chat.completions.calls[0]

    assert call["tools"][0]["function"]["name"] == "bash"
    assert call["parallel_tool_calls"] is False
    assert message["role"] == "user"
    assert message["content"] == (
        "You are continuing from a compacted earlier session.\n\n"
        "<compact_context>\n"
        "Keep src/a.py fix.\n"
        "</compact_context>"
    )
    assert message["extra"]["summary"] is True


def test_summarize_context_context_length_bad_request_raises_max_prompt_length():
    model = OpenAIClientModel(
        base_url="https://api.openai.com/v1",
        model_name="gpt-5-mini",
        api_key="test",
    )
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FailingCompletions("prompt length plus max_tokens exceeds limit")
        )
    )

    with pytest.raises(MaxPromptLengthExceeded):
        model.summarize_context([{"role": "user", "content": "old"}], "summarize")


def test_summarize_context_keeps_full_response_without_think_end():
    text = "<think>\nnot closed\nKeep src/a.py fix."
    response = FakeResponse(message=FakeMessage(content=text), finish_reason="stop")
    model = make_model(response)

    message = model.summarize_context([{"role": "user", "content": "old"}], "summarize")

    assert text in message["content"]


def test_observations_are_native_tool_messages_unless_raw_transcript():
    response = FakeResponse(message=FakeMessage(content=None, tool_calls=[tool_call()]))
    model = make_model(response)
    message = {
        "role": "assistant",
        "content": None,
        "extra": {"actions": [{"command": "pwd", "tool_call_id": "call_1"}]},
    }
    output = {"returncode": 0, "output": "ok", "exception_info": None}

    native = model.format_observation_messages(message, [output])
    raw = model.format_observation_messages(
        message | {"extra": message["extra"] | {"raw_transcript": True}},
        [output],
    )

    assert native[0]["role"] == "tool"
    assert native[0]["tool_call_id"] == "call_1"
    assert raw[0]["role"] == "user"
    assert "tool_call_id" not in raw[0]
    assert raw[0]["content"].startswith("<tool_response>")


def test_model_max_tokens_routes_to_model_kwargs():
    parser = argparse.ArgumentParser()
    add_flow_cli_args(parser)
    args = parser.parse_args(["--model_max_tokens", "123"])

    config = flow_config_from_args(args)

    assert config.model.as_model_config_overrides() == {"model_kwargs": {"max_tokens": 123}}


def test_model_sampling_overrides_route_to_model_kwargs():
    parser = argparse.ArgumentParser()
    add_flow_cli_args(parser)
    args = parser.parse_args(["--model_temperature", "1.0", "--model_top_p", "0.9"])

    overrides = flow_config_from_args(args).model.as_model_config_overrides()

    assert overrides == {"model_kwargs": {"temperature": 1.0, "top_p": 0.9}}


def test_model_return_token_ids_routes_to_model_kwargs():
    parser = argparse.ArgumentParser()
    add_flow_cli_args(parser)
    args = parser.parse_args(["--model_return_token_ids"])

    overrides = flow_config_from_args(args).model.as_model_config_overrides()

    assert overrides == {"model_kwargs": {"return_token_ids": True}}


@pytest.mark.parametrize(
    ("exit_status", "termination_reason"),
    [
        ("Submitted", TerminationReason.ENV_DONE),
        ("MaxPromptLengthExceeded", TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED),
        ("MaxResponseLengthExceeded", TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED),
        ("Timeout", TerminationReason.TIMEOUT),
        ("MaxTurnsExceeded", TerminationReason.MAX_TURNS_EXCEEDED),
    ],
)
def test_controlled_exit_statuses_classify_without_error_payload(exit_status, termination_reason):
    assert build_error_details(exit_status) is None
    assert classify_termination(exit_status) == termination_reason


def test_unknown_exit_status_classifies_as_error_with_error_payload():
    error = build_error_details("UnexpectedFailure")

    assert error == {
        "error_type": "error",
        "error_message": "UnexpectedFailure",
        "raw_error": "UnexpectedFailure",
    }
    assert classify_termination("UnexpectedFailure") == TerminationReason.ERROR


def _openai_model(*, base_url, model_name, model_kwargs=None, response=None):
    response = response or FakeResponse(message=FakeMessage(content=None, tool_calls=[tool_call()]))
    model = OpenAIClientModel(
        base_url=base_url,
        model_name=model_name,
        api_key="test",
        model_config={"model_kwargs": model_kwargs} if model_kwargs else None,
    )
    model.client = FakeClient(response)
    return model


def test_official_openai_strips_extra_body():
    model = _openai_model(
        base_url="https://api.openai.com/v1",
        model_name="gpt-5-mini",
        model_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
    )
    model.query([{"role": "user", "content": "hi"}])
    assert "extra_body" not in model.client.chat.completions.calls[0]


def test_gpt5_drops_temperature():
    model = _openai_model(
        base_url="https://api.openai.com/v1",
        model_name="gpt-5-mini",
        model_kwargs={"temperature": 0.0},
    )
    model.query([{"role": "user", "content": "hi"}])
    assert "temperature" not in model.client.chat.completions.calls[0]


def test_unknown_parameter_400_propagates():
    from openai import BadRequestError

    class FailingCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise BadRequestError(
                message="Error code: 400 - {'error': {'message': \"Unknown parameter: 'foo'.\", 'code': 'unknown_parameter'}}",
                response=SimpleNamespace(request=None, status_code=400, headers={}),
                body={"error": {"code": "unknown_parameter"}},
            )

    model = OpenAIClientModel(
        base_url="https://api.openai.com/v1",
        model_name="gpt-5-mini",
        api_key="test",
    )
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    with pytest.raises(BadRequestError):
        model.query([{"role": "user", "content": "hi"}])


# --- vLLM / Qwen3.5 via local rLLM gateway: production-shape integration tests ---
# Response shape comes from rllm parser transport: choices[0].token_ids,
# choices[0].message with tool_calls, finish_reason="tool_calls". Base URL points
# at a non-OpenAI gateway so chat_template_kwargs flows through extra_body.

_VLLM_BASE_URL = "http://gateway.local/v1"
_QWEN_MODEL = "Qwen/Qwen3.5-9B"
_QWEN_TOKENIZER_CANDIDATES = (
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-2B",
)


@pytest.fixture(scope="module")
def qwen35_tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        pytest.skip(f"transformers AutoTokenizer unavailable: {e}")

    last_error = None
    for name in _QWEN_TOKENIZER_CANDIDATES:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            last_error = e
            continue
        if tokenizer.eos_token_id is None:
            pytest.skip(f"{name} tokenizer has no eos_token_id")
        return name, tokenizer

    pytest.skip(f"No local Qwen3.5 tokenizer cache found: {last_error}")


def _decode(tokenizer, token_ids):
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def test_vllm_qwen35_full_query_path():
    response = FakeResponse(
        message=FakeMessage(content="raw bytes ignored", tool_calls=[tool_call()]),
        token_ids=[10, 11, 12, FakeTokenizer.eos_token_id],
    )
    model = make_model(
        response,
        base_url=_VLLM_BASE_URL,
        model_name=_QWEN_MODEL,
        model_config={
            "model_kwargs": {
                "temperature": 0.0,
                "parallel_tool_calls": False,
                "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        },
    )
    model.tokenizer = FakeTokenizer()

    message = model.query([{"role": "user", "content": "fix the bug"}])

    call = model.client.chat.completions.calls[0]
    assert call["temperature"] == 0.0
    assert call["parallel_tool_calls"] is False
    assert call["extra_body"]["return_token_ids"] is True
    assert call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert call["tools"][0]["function"]["name"] == "bash"

    assert message["content"] == "10 11 12"
    assert "tool_calls" not in message
    assert message["extra"]["raw_transcript"] is True
    assert message["extra"]["completion_token_ids"] == [10, 11, 12, 99]
    assert message["extra"]["actions"] == [{"command": "pwd", "tool_call_id": "call_1"}]


def test_vllm_multi_turn_sends_decoded_content_back():
    first = FakeResponse(
        message=FakeMessage(content=None, tool_calls=[tool_call()]),
        token_ids=[1, 2, FakeTokenizer.eos_token_id],
    )
    second = FakeResponse(
        message=FakeMessage(content=None, tool_calls=[tool_call()]),
        token_ids=[3, 4, FakeTokenizer.eos_token_id],
    )
    model = OpenAIClientModel(
        base_url=_VLLM_BASE_URL,
        model_name=_QWEN_MODEL,
        api_key="test",
        model_config={"model_kwargs": {"temperature": 0.0, "return_token_ids": True}},
    )
    model.client = FakeClient(first, second)
    model.tokenizer = FakeTokenizer()

    turn1 = model.query([{"role": "user", "content": "hi"}])
    history = [{"role": "user", "content": "hi"}, turn1]
    model.query(history)

    second_call_messages = model.client.chat.completions.calls[1]["messages"]
    assistant_msg = second_call_messages[1]
    assert assistant_msg == {"role": "assistant", "content": "1 2"}


def test_vllm_token_aware_format_error_keeps_decoded_text():
    response = FakeResponse(
        message=FakeMessage(content=None, tool_calls=[tool_call(arguments="{bad json")]),
        token_ids=[20, 21, FakeTokenizer.eos_token_id],
    )
    model = make_model(
        response,
        base_url=_VLLM_BASE_URL,
        model_name=_QWEN_MODEL,
        model_config={"model_kwargs": {"temperature": 0.0, "return_token_ids": True}},
    )
    model.tokenizer = FakeTokenizer()

    message = model.query([{"role": "user", "content": "hi"}])

    assert message["content"] == "20 21"
    assert "tool_calls" not in message
    assert message["extra"]["actions"] == []
    assert message["extra"]["raw_transcript"] is True
    assert "format_error" in message["extra"]


def test_vllm_qwen35_local_tokenizer_decodes_raw_transcript(qwen35_tokenizer):
    tokenizer_name, tokenizer = qwen35_tokenizer
    assert tokenizer.eos_token == "<|im_end|>"

    completion_text = (
        "<think>\n"
        "keep this reasoning across turns\n"
        "</think>\n"
        "<tool_call>\n"
        '{"name": "bash", "arguments": {"command": "pwd"}}\n'
        "</tool_call>"
    )
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    token_ids = completion_ids + [tokenizer.eos_token_id]
    response = FakeResponse(
        message=FakeMessage(content="gateway parsed text", tool_calls=[tool_call()]),
        token_ids=token_ids,
    )
    model = make_model(
        response,
        base_url=_VLLM_BASE_URL,
        model_name=tokenizer_name,
        model_config={
            "tokenizer_name": tokenizer_name,
            "model_kwargs": {
                "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        },
    )

    message = model.query([{"role": "user", "content": "fix the bug"}])

    assert message["content"] == _decode(tokenizer, completion_ids)
    assert message["extra"]["completion_token_ids"] == token_ids
    assert message["extra"]["raw_transcript"] is True
    assert "<think>" in message["content"]
    assert "<tool_call>" in message["content"]
    assert "tool_calls" not in message
    assert message["extra"]["actions"] == [{"command": "pwd", "tool_call_id": "call_1"}]

    call = model.client.chat.completions.calls[0]
    assert call["extra_body"]["return_token_ids"] is True
    assert call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_vllm_qwen35_local_tokenizer_length_finish_reason_raises(qwen35_tokenizer):
    tokenizer_name, tokenizer = qwen35_tokenizer
    token_ids = tokenizer.encode("<think>\nunfinished", add_special_tokens=False)
    assert tokenizer.eos_token_id not in token_ids

    response = FakeResponse(
        message=FakeMessage(content="partial", tool_calls=[tool_call()]),
        finish_reason="length",
        token_ids=token_ids,
    )
    model = make_model(
        response,
        base_url=_VLLM_BASE_URL,
        model_name=tokenizer_name,
        model_config={
            "tokenizer_name": tokenizer_name,
            "model_kwargs": {"return_token_ids": True},
        },
    )

    with pytest.raises(MaxResponseLengthExceeded):
        model.query([{"role": "user", "content": "fix the bug"}])


def make_progress_agent(model, **kwargs):
    agent_kwargs = {
        "system_template": "{{task}}",
        "instance_template": "{{task}}",
        "step_limit": 10,
    }
    agent_kwargs.update(kwargs)
    return ProgressLoggingAgent(
        model=model,
        env=DummyEnv(),
        log_fn=lambda _: None,
        **agent_kwargs,
    )


def test_progress_agent_converts_max_response_length_to_limits_exceeded():
    agent = make_progress_agent(MaxResponseModel())

    with pytest.raises(LimitsExceeded) as exc:
        agent.step()

    assert exc.value.messages[0]["content"] == "MaxResponseLengthExceeded"
    assert exc.value.messages[0]["extra"]["exit_status"] == "MaxResponseLengthExceeded"


def test_progress_agent_converts_max_prompt_length_to_limits_exceeded():
    agent = make_progress_agent(MaxPromptModel())

    with pytest.raises(LimitsExceeded) as exc:
        agent.step()

    assert exc.value.messages[0]["content"] == "MaxPromptLengthExceeded"
    assert exc.value.messages[0]["extra"]["exit_status"] == "MaxPromptLengthExceeded"


def test_progress_agent_timeout_uses_canonical_exit_status():
    agent = make_progress_agent(MaxResponseModel(), agent_timeout=1)
    agent._start_time = time.monotonic() - 2

    with pytest.raises(LimitsExceeded) as exc:
        agent.step()

    assert exc.value.messages[0]["content"] == "Timeout"
    assert exc.value.messages[0]["extra"]["exit_status"] == "Timeout"


def test_progress_agent_max_turns_uses_canonical_exit_status():
    agent = make_progress_agent(MaxResponseModel(), step_limit=1)
    agent.n_calls = 1

    with pytest.raises(LimitsExceeded) as exc:
        agent.step()

    assert exc.value.messages[0]["content"] == "MaxTurnsExceeded"
    assert exc.value.messages[0]["extra"]["exit_status"] == "MaxTurnsExceeded"


def test_progress_agent_run_records_max_response_length_without_traceback():
    agent = make_progress_agent(MaxResponseModel())

    result = agent.run("task")

    assert result["exit_status"] == "MaxResponseLengthExceeded"
    assert agent.messages[-1]["content"] == "MaxResponseLengthExceeded"
    assert "traceback" not in agent.messages[-1].get("extra", {})
