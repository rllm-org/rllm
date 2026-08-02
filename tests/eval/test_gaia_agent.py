"""Unit tests for the GAIA agent's tool-calling loop, with a fake client + fake
tool so the control flow is covered without network or API keys.
"""

from __future__ import annotations

from types import SimpleNamespace

from cookbooks.gaia.agent import run_tool_loop  # repo root on sys.path via conftest.py


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


class _RaisingTool:
    """Tool whose execution raises (missing API key, bad kwargs, network error)."""

    def __init__(self, exc):
        self.json = {
            "type": "function",
            "function": {"name": "tavily-search", "description": "search", "parameters": {"type": "object", "properties": {}}},
        }
        self._exc = exc

    def forward(self, **args):
        raise self._exc


def test_tool_exception_becomes_observation_not_crash():
    """A raising tool must yield an error observation, not kill the rollout."""
    client = _Client(
        [
            _Msg(tool_calls=[_ToolCall("tavily-search", '{"query": "x"}')]),
            _Msg(content="FINAL ANSWER: recovered"),
        ]
    )
    steps, answer = run_tool_loop(client, "test-model", [_RaisingTool(ValueError("TAVILY_API_KEY is not set"))], "q", max_turns=5)
    assert answer == "FINAL ANSWER: recovered"
    assert any("Tool error" in (s.output or "") for s in steps)


def test_invalid_json_arguments_do_not_crash():
    """Malformed tool arguments fall back to {} — and a TypeError from the tool
    (missing required kwarg) is caught as an observation."""
    client = _Client(
        [
            _Msg(tool_calls=[_ToolCall("tavily-search", "not-valid-json{{")]),
            _Msg(content="FINAL ANSWER: ok"),
        ]
    )
    steps, answer = run_tool_loop(client, "test-model", [_RaisingTool(TypeError("missing required argument: 'query'"))], "q", max_turns=5)
    assert answer == "FINAL ANSWER: ok"
    assert any("Tool error" in (s.output or "") for s in steps)


def test_parallel_tool_calls_record_one_step_per_turn():
    """Regression: the trace enricher consumes traces 1:1 with steps (one LLM call ==
    one Step), so N parallel tool_calls in one turn must still record exactly one Step."""
    tool = _FakeSearchTool()
    client = _Client(
        [
            _Msg(tool_calls=[_ToolCall("tavily-search", '{"query": "a"}', id="tc1"), _ToolCall("tavily-search", '{"query": "b"}', id="tc2")]),
            _Msg(content="FINAL ANSWER: both"),
        ]
    )
    steps, answer = run_tool_loop(client, "test-model", [tool], "q", max_turns=5)
    assert answer == "FINAL ANSWER: both"
    assert len(steps) == 2  # one tool turn + one answer turn
    assert len(tool.calls) == 2  # but both tool calls executed


def test_max_turns_exhaustion_marks_last_step_done():
    """Exhaustion terminates the episode: last step done=True (flag flipped, never
    appended — parity with traces), answer stays empty."""
    tool = _FakeSearchTool()
    msgs = [_Msg(tool_calls=[_ToolCall("tavily-search", '{"query": "again"}')]) for _ in range(3)]
    client = _Client(msgs)
    steps, answer = run_tool_loop(client, "test-model", [tool], "q", max_turns=3)
    assert answer == ""
    assert len(steps) == 3
    assert steps[-1].done is True
    assert all(not s.done for s in steps[:-1])


def test_llm_error_records_no_step():
    """A failed LLM call must not append a Step: no gateway trace exists for it and
    the enricher requires step<->trace parity."""

    class _FailingClient:
        def __init__(self):
            def _create(**_kwargs):
                raise RuntimeError("connection reset")

            from types import SimpleNamespace

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    steps, answer = run_tool_loop(_FailingClient(), "test-model", [_FakeSearchTool()], "q", max_turns=5)
    assert steps == []
    assert answer == ""
