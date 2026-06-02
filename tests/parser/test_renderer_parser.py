"""Integration tests for RendererParser and the multi-turn ParserSession.

These exercise the real ``renderers`` package against a real tokenizer
(tokenizer files only — no model weights), so they are skipped when the
optional ``renderers`` dependency is not installed.
"""

import pytest

pytest.importorskip("renderers")

from transformers import AutoTokenizer

from rllm.parser.base import BaseParser, ParsedCompletion, ParserSession
from rllm.parser.renderer_parser import RendererParser
from rllm.tools.tool_base import ToolCall

MODEL = "Qwen/Qwen3-0.6B"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def parser(tokenizer):
    return RendererParser.from_tokenizer(tokenizer)


def test_implements_base_parser(parser):
    assert isinstance(parser, BaseParser)
    # auto-detection should pick the hand-coded Qwen3 renderer
    assert parser.renderer_name == "Qwen3Renderer"


def test_render_returns_token_ids(parser):
    ids = parser.render([{"role": "user", "content": "hello"}], add_generation_prompt=True)
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_generation_prompt_extends_render(parser):
    messages = [{"role": "user", "content": "hello"}]
    without = parser.render(messages, add_generation_prompt=False)
    with_prompt = parser.render(messages, add_generation_prompt=True)
    assert len(with_prompt) > len(without)
    assert with_prompt[: len(without)] == without


def test_get_stop_token_ids(parser):
    stop_ids = parser.get_stop_token_ids()
    assert isinstance(stop_ids, list)
    assert len(stop_ids) > 0
    assert all(isinstance(i, int) for i in stop_ids)


def test_parse_completion_content_and_reasoning(parser, tokenizer):
    completion = tokenizer.encode("<think>thinking it through</think>The answer is 42.", add_special_tokens=False)
    parsed = parser.parse_completion(completion)
    assert isinstance(parsed, ParsedCompletion)
    assert parsed.content == "The answer is 42."
    assert parsed.reasoning == "thinking it through"
    assert parsed.tool_calls == []


def test_parse_completion_tool_calls(parser):
    """Render an assistant tool call, slice off the completion, parse it back."""
    user = [{"role": "user", "content": "weather in NYC?"}]
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning": "need to call the tool",
        "tool_calls": [ToolCall(name="get_weather", arguments={"city": "NYC"})],
    }
    prompt_ids = parser.render(user, tools=[WEATHER_TOOL], add_generation_prompt=True)
    full_ids = parser.render(user + [assistant], tools=[WEATHER_TOOL], add_generation_prompt=False)
    completion_ids = full_ids[len(prompt_ids) :]

    parsed = parser.parse_completion(completion_ids)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.name == "get_weather"
    assert call.arguments == {"city": "NYC"}
    assert parsed.reasoning == "need to call the tool"


def test_bridge_extends_previous_turn(parser):
    user = [{"role": "user", "content": "weather in NYC?"}]
    prompt_ids = parser.render(user, tools=[WEATHER_TOOL], add_generation_prompt=True)
    completion_ids = parser.render(
        user + [{"role": "assistant", "content": "Let me check.", "reasoning": "ok"}],
        tools=[WEATHER_TOOL],
    )[len(prompt_ids) :]

    new_messages = [{"role": "tool", "content": "sunny, 75F", "name": "get_weather"}]
    bridged = parser.bridge_to_next_turn(prompt_ids, completion_ids, new_messages, tools=[WEATHER_TOOL])

    assert bridged is not None
    prefix = prompt_ids + completion_ids
    # Contract: the bridge result reuses the prior tokens verbatim.
    assert bridged[: len(prefix)] == prefix
    assert len(bridged) > len(prefix)


def test_bridge_matches_full_render(parser):
    """The bridge output must equal a full render of the same conversation."""
    user = [{"role": "user", "content": "weather in NYC?"}]
    assistant = {"role": "assistant", "content": "Let me check.", "reasoning": "ok"}
    tool_msg = {"role": "tool", "content": "sunny, 75F", "name": "get_weather"}

    prompt_ids = parser.render(user, tools=[WEATHER_TOOL], add_generation_prompt=True)
    completion_ids = parser.render(user + [assistant], tools=[WEATHER_TOOL])[len(prompt_ids) :]

    bridged = parser.bridge_to_next_turn(prompt_ids, completion_ids, [tool_msg], tools=[WEATHER_TOOL])
    full = parser.render(user + [assistant, tool_msg], tools=[WEATHER_TOOL], add_generation_prompt=True)
    assert bridged == full


def test_parser_session_multiturn_uses_bridge(parser):
    session = parser.new_session(tools=[WEATHER_TOOL])
    assert isinstance(session, ParserSession)

    prompt_ids = session.start([{"role": "user", "content": "weather in NYC?"}])
    assert len(prompt_ids) > 0

    # turn 0: model emits an assistant turn
    assistant = {"role": "assistant", "content": "Checking.", "reasoning": "ok"}
    completion_ids = parser.render(session.messages + [assistant], tools=[WEATHER_TOOL])[len(prompt_ids) :]
    session.observe_completion(completion_ids, assistant_message=assistant)

    # turn 1: environment returns a tool result; advance via the bridge
    next_prompt = session.advance([{"role": "tool", "content": "sunny, 75F", "name": "get_weather"}])
    assert next_prompt[: len(prompt_ids) + len(completion_ids)] == prompt_ids + completion_ids
    assert session.stats == {"turns": 1, "bridged": 1, "rerendered": 0}

    # the advanced prompt matches a fresh full render of the conversation
    full = parser.render(session.messages, tools=[WEATHER_TOOL], add_generation_prompt=True)
    assert next_prompt == full


def test_default_renderer_has_no_bridge(tokenizer):
    """DefaultRenderer can't prove the extension is safe, so the session
    falls back to full re-renders."""
    parser = RendererParser.from_tokenizer(tokenizer, renderer="default")
    assert parser.renderer_name == "DefaultRenderer"

    session = parser.new_session()
    prompt_ids = session.start([{"role": "user", "content": "hi"}])
    completion_ids = parser.render(session.messages + [{"role": "assistant", "content": "hello"}])[len(prompt_ids) :]
    session.observe_completion(completion_ids, assistant_message={"role": "assistant", "content": "hello"})

    session.advance([{"role": "user", "content": "and again"}])
    assert session.stats == {"turns": 1, "bridged": 0, "rerendered": 1}


def test_session_lifecycle_errors(parser):
    session = parser.new_session()
    with pytest.raises(RuntimeError):
        session.advance([{"role": "user", "content": "x"}])

    session.start([{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError):
        session.start([{"role": "user", "content": "again"}])
