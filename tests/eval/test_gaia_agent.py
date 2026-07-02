"""Unit tests for the GAIA agent's tool-calling loop, with a fake client + fake
tool so the control flow is covered without network or API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `cookbooks` import

from cookbooks.gaia.agent import run_tool_loop  # noqa: E402


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments, id="tc0"):
        self.function = _Fn(name, arguments)
        self.id = id


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=False):
        d = {"role": "assistant", "content": self.content}
        return {k: v for k, v in d.items() if v is not None} if exclude_none else d


class _Client:
    """Returns the scripted messages one per create() call."""

    def __init__(self, msgs):
        self._msgs = list(msgs)
        self.create_calls = 0

        def _create(**_kwargs):
            self.create_calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=self._msgs.pop(0))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


class _FakeSearchTool:
    def __init__(self):
        self.json = {
            "type": "function",
            "function": {"name": "tavily-search", "description": "search", "parameters": {"type": "object", "properties": {}}},
        }
        self.calls = []

    def forward(self, **args):
        self.calls.append(args)
        return SimpleNamespace(error=None, output={"results": [{"title": "Paris", "content": "capital of France"}]})


def test_loop_calls_tool_then_answers():
    tool = _FakeSearchTool()
    client = _Client(
        [
            _Msg(tool_calls=[_ToolCall("tavily-search", '{"query": "capital of France"}')]),
            _Msg(content="FINAL ANSWER: Paris"),
        ]
    )
    steps, answer = run_tool_loop(client, "test-model", [tool], "What is the capital of France?", max_turns=5)

    assert answer == "FINAL ANSWER: Paris"
    assert tool.calls == [{"query": "capital of France"}]
    assert client.create_calls == 2
    assert steps[-1].done is True
    # the search observation made it into the transcript
    assert any("capital of France" in (s.output or "") for s in steps)


def test_loop_answers_without_tool_call():
    client = _Client([_Msg(content="FINAL ANSWER: 4")])
    steps, answer = run_tool_loop(client, "test-model", [_FakeSearchTool()], "2+2?", max_turns=3)
    assert "4" in answer
    assert client.create_calls == 1


def test_loop_handles_unknown_tool():
    client = _Client(
        [
            _Msg(tool_calls=[_ToolCall("nonexistent", "{}")]),
            _Msg(content="FINAL ANSWER: done"),
        ]
    )
    steps, answer = run_tool_loop(client, "test-model", [_FakeSearchTool()], "q", max_turns=5)
    assert answer == "FINAL ANSWER: done"
    assert any("Unknown tool" in (s.output or "") for s in steps)
