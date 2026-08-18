from types import SimpleNamespace

from rllm_model_gateway.v2.tokenization import (
    TokenizationService,
    _to_api_tool_calls,
    _to_renderer_tools,
)


def test_openai_function_tools_are_converted_for_renderer() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find a value",
                "parameters": {"type": "object"},
            },
        }
    ]

    assert _to_renderer_tools(tools) == [
        {
            "name": "lookup",
            "description": "Find a value",
            "parameters": {"type": "object"},
        }
    ]
    assert _to_renderer_tools([]) is None


def test_parsed_tool_calls_are_converted_to_openai_shape() -> None:
    parsed = [
        SimpleNamespace(id="call-1", name="lookup", arguments={"query": "value"}),
        SimpleNamespace(id=None, name="other", arguments='{"value":1}'),
        SimpleNamespace(id="ignored", name="", arguments={}),
    ]

    calls = _to_api_tool_calls(parsed)  # type: ignore[arg-type]

    assert calls[0] == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query":"value"}'},
    }
    assert calls[1]["id"].startswith("call_")
    assert calls[1]["function"] == {"name": "other", "arguments": '{"value":1}'}
    assert len(calls) == 2


class FakeTokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1, 2]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "decoded"


class FakeRenderer:
    def __init__(self) -> None:
        self.bridge_result = SimpleNamespace(token_ids=[1, 2, 3])

    def render_ids(self, messages, *, tools, add_generation_prompt):
        assert add_generation_prompt
        assert tools is None
        return [3, 4]

    def bridge_to_next_turn(self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools):
        assert previous_prompt_ids == [1]
        assert previous_completion_ids == [2]
        assert new_messages == [{"role": "user", "content": "next"}]
        assert tools is None
        return self.bridge_result

    def parse_response(self, token_ids, *, tools):
        assert token_ids == [5]
        assert tools is None
        return SimpleNamespace(
            content="answer",
            reasoning_content="reasoning",
            tool_calls=[SimpleNamespace(id="call-1", name="lookup", arguments={})],
        )

    def get_stop_token_ids(self):
        return (98, 99)


def _service() -> TokenizationService:
    service = TokenizationService.__new__(TokenizationService)
    service._tokenizer = FakeTokenizer()
    service._renderer = FakeRenderer()
    return service


def test_tokenization_service_delegates_render_parse_and_decode() -> None:
    service = _service()

    assert service.encode("prompt") == [1, 2]
    assert service.decode([1, 2]) == "decoded"
    assert service.render([{"role": "user", "content": "question"}], []) == [3, 4]
    assert service.bridge([1], [2], [{"role": "user", "content": "next"}], []) == [1, 2, 3]
    assert service.parse_completion([5], []) == {
        "content": "answer",
        "reasoning_content": "reasoning",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    assert service.stop_token_ids() == [98, 99]


def test_tokenization_service_preserves_renderer_bridge_failure() -> None:
    service = _service()
    service._renderer.bridge_result = None

    assert service.bridge([1], [2], [{"role": "user", "content": "next"}], []) is None
