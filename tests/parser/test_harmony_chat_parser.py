import pytest

from rllm.parser.chat_template_parser import HarmonyChatTemplateParser
from rllm.tools.tool_base import Tool, ToolCall, ToolOutput


class BuiltinTool:
    name = "python"


@pytest.fixture
def harmony_parser(tmp_path, monkeypatch):
    pytest.importorskip("openai_harmony")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    try:
        return HarmonyChatTemplateParser()
    except Exception as exc:
        pytest.skip(f"Harmony encoding unavailable: {exc}")


def lookup(query: str) -> str:
    """Look up a query."""
    return query


def completion_tokens(parser: HarmonyChatTemplateParser, message) -> list[int]:
    rendered = parser.enc.decode_utf8(parser.enc.render(message))
    completion = rendered.removeprefix("<|start|>assistant")
    return parser.enc.encode(completion, allowed_special="all")


def test_harmony_renders_builtin_and_function_tools(harmony_parser):
    prompt = harmony_parser.parse(
        [{"role": "user", "content": "use tools"}],
        is_first_msg=True,
        tools=[BuiltinTool(), Tool(function=lookup)],
    )

    assert "## python" in prompt
    assert "namespace functions" in prompt
    assert "type lookup" in prompt


def test_harmony_renders_assistant_tool_calls_and_tool_messages(harmony_parser):
    prompt = harmony_parser.parse(
        [
            {"role": "user", "content": "look up weather"},
            {
                "role": "assistant",
                "tool_calls": [ToolCall(name="lookup", arguments={"query": "weather"})],
            },
            {
                "role": "tool",
                "tool_outputs": [ToolOutput(name="lookup", output={"result": "sunny"})],
            },
            {"role": "assistant", "content": "It is sunny."},
        ],
        is_first_msg=True,
        tools=[Tool(function=lookup)],
    )

    assert "to=functions.lookup" in prompt
    assert '{"query": "weather"}' in prompt
    assert "<|start|>functions.lookup to=assistant" in prompt
    assert '{"result": "sunny"}' in prompt
    assert "It is sunny." in prompt


def test_harmony_renders_builtin_python_tool_call(harmony_parser):
    prompt = harmony_parser.parse(
        [
            {"role": "user", "content": "compute"},
            {
                "role": "assistant",
                "tool_calls": [ToolCall(name="python", arguments={"code": "print(1)"})],
            },
        ],
        is_first_msg=True,
        tools=[BuiltinTool()],
    )

    assert "to=python" in prompt
    assert "analysis code" in prompt
    assert "print(1)" in prompt


def test_harmony_parse_completion_tool_calls(harmony_parser):
    from openai_harmony import Message, Role

    function_message = Message.from_role_and_content(Role.ASSISTANT, '{"query":"weather"}').with_channel("commentary").with_recipient("functions.lookup").with_content_type("json")
    parsed = harmony_parser.parse_completion(completion_tokens(harmony_parser, function_message))

    assert parsed["content"] == ""
    assert parsed["reasoning"] == ""
    assert parsed["tool_calls"] == [ToolCall(name="lookup", arguments={"query": "weather"})]

    python_message = Message.from_role_and_content(Role.ASSISTANT, "print(1)").with_channel("analysis").with_recipient("python").with_content_type("code")
    parsed = harmony_parser.parse_completion(completion_tokens(harmony_parser, python_message))

    assert parsed["tool_calls"] == [ToolCall(name="python", arguments={"code": "print(1)"})]
