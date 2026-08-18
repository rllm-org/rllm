import json

import pytest
from rllm_model_gateway.v2.protocols import normalize_request, response_payload, stream_events
from rllm_model_gateway.v2.types import APIProtocol, GatewayError, GatewayResponse


def _response(**overrides) -> GatewayResponse:
    values = {
        "request_id": "req_1",
        "text": "raw text",
        "content": "answer",
        "reasoning_content": "reasoning",
        "tool_calls": [],
        "finish_reason": "stop",
        "prompt_tokens": 4,
        "completion_tokens": 2,
    }
    values.update(overrides)
    return GatewayResponse(**values)


def test_chat_normalization_separates_protocol_and_sampling_fields() -> None:
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

    request = normalize_request(
        APIProtocol.CHAT_COMPLETIONS,
        "session-1",
        {
            "model": "model",
            "messages": messages,
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
            "user": "ignored-routing-hint",
            "temperature": 0.7,
            "max_tokens": 20,
            "n": 1,
        },
    )

    assert request.request_id.startswith("req_")
    assert request.session_id == "session-1"
    assert request.messages == messages
    assert request.tools == tools
    assert request.sampling_params == {"temperature": 0.7, "max_tokens": 20, "n": 1}


@pytest.mark.parametrize(
    ("prompt", "expected_prompt", "expected_token_ids"),
    [
        ("hello", "hello", None),
        ([1, 2, 3], None, [1, 2, 3]),
    ],
)
def test_completion_normalization_accepts_text_or_token_ids(prompt, expected_prompt, expected_token_ids) -> None:
    request = normalize_request(
        APIProtocol.COMPLETIONS,
        "session-1",
        {"model": "model", "prompt": prompt, "temperature": 0.2},
    )

    assert request.prompt == expected_prompt
    assert request.prompt_token_ids == expected_token_ids
    assert request.messages == []
    assert request.tools == []
    assert request.sampling_params == {"temperature": 0.2}


@pytest.mark.parametrize("prompt", [None, [1, True], [1, "2"], [1.0, 2.0], ["one", "two"]])
def test_completion_normalization_rejects_unsupported_prompt_shapes(prompt) -> None:
    with pytest.raises(GatewayError, match="one string or token-id prompt"):
        normalize_request(APIProtocol.COMPLETIONS, "session-1", {"prompt": prompt})


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"messages": []}, "non-empty array"),
        ({"messages": ["hello"]}, "object with a role"),
        ({"messages": [{"content": "hello"}]}, "object with a role"),
        ({"messages": [{"role": "user"}], "tools": {}}, "tools must be an array"),
        (
            {"messages": [{"role": "user"}], "tools": [{"type": "function", "function": {}}]},
            "name must be a non-empty string",
        ),
    ],
)
def test_chat_normalization_validates_messages_and_tools(body, message) -> None:
    with pytest.raises(GatewayError, match=message):
        normalize_request(APIProtocol.CHAT_COMPLETIONS, "session-1", body)


def test_chat_response_payload_preserves_reasoning_and_tool_calls() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    payload = response_payload(
        APIProtocol.CHAT_COMPLETIONS,
        _response(tool_calls=tool_calls, finish_reason="tool_calls"),
        "model",
        123,
    )

    assert payload["id"] == "req_1"
    assert payload["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "reasoning",
                "tool_calls": tool_calls,
            },
            "logprobs": None,
            "finish_reason": "tool_calls",
        }
    ]
    assert payload["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}


def test_completion_response_uses_decoded_text() -> None:
    payload = response_payload(APIProtocol.COMPLETIONS, _response(), "model", 123)

    assert payload["choices"] == [{"index": 0, "text": "raw text", "logprobs": None, "finish_reason": "stop"}]


@pytest.mark.parametrize("protocol", [APIProtocol.CHAT_COMPLETIONS, APIProtocol.COMPLETIONS])
def test_stream_events_emit_response_usage_and_done(protocol: APIProtocol) -> None:
    events = list(stream_events(protocol, _response(), "model", 123, include_usage=True))

    assert events[-1] == "data: [DONE]\n\n"
    decoded = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert decoded[-1]["choices"] == []
    assert decoded[-1]["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }
    assert all("usage" in event for event in decoded)
