"""Terminus-2-native harness: the tool-calling conversation shape.

The whole point of this harness is the message-loop mechanics — native
``tool_calls`` + ``role:tool`` results + ``reasoning_content`` echoed on
assistant messages, so think-mode models retain reasoning across turns
(measured: the tool-calling context path renders echoed reasoning into the
prompt; the user-message path silently drops it). The in-sandbox driver is
pure Python, so these tests exec it in-process and drive its loop with a
scripted LLM and a fake tmux session.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import types

from rllm.harnesses.terminus2 import _DRIVER_PATH
from rllm.harnesses.terminus2_native import _NATIVE_DRIVER_SCRIPT, Terminus2NativeHarness

# ---------------------------------------------------------------------------
# Driver module loaded in-process (harbor + httpx are host-venv deps)
# ---------------------------------------------------------------------------

_ENV_DEFAULTS = {
    "RLLM_TERMINUS_MODEL": "openai/accounts/fireworks/models/deepseek-v4-pro",
    "RLLM_TERMINUS_API_BASE": "http://gw:9090/sessions/t:0/v1",
    "RLLM_TERMINUS_TEMPERATURE": "1.0",
}


def _load_driver_module() -> types.ModuleType:
    for k, v in _ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    mod = types.ModuleType("terminus2_native_driver")
    exec(compile(_NATIVE_DRIVER_SCRIPT, "<driver>", "exec"), mod.__dict__)
    return mod


class _FakeTmux:
    """Stands in for harbor's TmuxSession: records keystrokes, serves screens."""

    def __init__(self, *args, **kwargs):
        self.sent: list[tuple[str, float]] = []
        self.screens = ["initial screen"]

    async def start(self):
        return None

    async def send_keys(self, keys, block=False, min_timeout_sec=1.0):
        self.sent.append((keys, min_timeout_sec))

    async def get_incremental_output(self):
        return self.screens.pop(0) if self.screens else "screen"

    async def capture_pane(self, capture_entire=False):
        return "captured screen"


def _tool_call(arguments: dict, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": "execute_commands", "arguments": json.dumps(arguments)}}


def _run_driver(llm_script: list[dict], max_steps: int = 500):
    """Run the driver loop against a scripted LLM; return (captured_requests, tmux)."""
    mod = _load_driver_module()
    tmux = _FakeTmux()
    mod.TmuxSession = lambda *a, **kw: tmux

    driver = mod.Driver()
    driver.max_steps = max_steps
    requests: list[list[dict]] = []
    responses = iter(llm_script)

    async def fake_llm(client, messages, use_tools=True, tool_choice=None):
        requests.append(json.loads(json.dumps(messages)))  # deep snapshot
        driver.n_calls += 1
        return next(responses)

    driver.llm = fake_llm
    asyncio.run(driver.run("solve the task"))
    return requests, tmux, driver


# ---------------------------------------------------------------------------
# The load-bearing mechanics
# ---------------------------------------------------------------------------


def test_reasoning_content_is_echoed_back_on_assistant_messages():
    """THE feature: reasoning from turn N must appear on the assistant message
    the model sees at turn N+1 — on the tool-calling path, templates render it."""
    script = [
        {"content": "", "tool_calls": [_tool_call({"commands": [{"keystrokes": "ls\n", "duration": 1}]})], "reasoning_content": "THINKING-TURN-1"},
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True}, "call_2")]},
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True}, "call_3")]},
    ]
    requests, tmux, _ = _run_driver(script)

    assert len(requests) == 3
    second = requests[1]
    assistants = [m for m in second if m["role"] == "assistant"]
    assert assistants[0].get("reasoning_content") == "THINKING-TURN-1"
    assert assistants[0]["tool_calls"][0]["function"]["name"] == "execute_commands"
    # terminal output came back as a tool message tied to the call id
    tools = [m for m in second if m["role"] == "tool"]
    assert tools and tools[0]["tool_call_id"] == "call_1"
    assert tmux.sent == [("ls\n", 1.0)]


def test_task_complete_requires_confirmation_then_ends():
    script = [
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True})]},
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True}, "call_2")]},
    ]
    requests, _, driver = _run_driver(script)
    assert driver.n_calls == 2  # asked once to confirm, then stopped
    confirm_prompt = [m for m in requests[1] if m["role"] == "tool"][-1]["content"]
    assert "task_complete=true" in confirm_prompt


def test_task_complete_retracted_resets_confirmation():
    script = [
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True})]},
        {"content": "", "tool_calls": [_tool_call({"commands": [{"keystrokes": "make\n"}]}, "call_2")]},  # keeps working
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True}, "call_3")]},
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True}, "call_4")]},
    ]
    _, _, driver = _run_driver(script)
    assert driver.n_calls == 4  # retraction meant a fresh confirm round was needed


def test_three_no_tool_turns_end_the_run():
    script = [
        {"content": "just chatting"},
        {"content": "still chatting"},
        {"content": "no tools ever"},
    ]
    requests, _, driver = _run_driver(script)
    assert driver.n_calls == 3
    nudges = [m for m in requests[-1] if m["role"] == "user" and "execute_commands" in m.get("content", "")]
    assert nudges, "driver should nudge the model back toward the tool"


def test_malformed_tool_arguments_surface_as_tool_error():
    bad = {"id": "call_x", "type": "function", "function": {"name": "execute_commands", "arguments": "{not json"}}
    script = [
        {"content": "", "tool_calls": [bad]},
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True}, "c2")]},
        {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True}, "c3")]},
    ]
    requests, _, _ = _run_driver(script)
    err = [m for m in requests[1] if m["role"] == "tool"][0]
    assert "could not parse tool arguments" in err["content"]


def test_history_budget_truncates_oldest_tool_output_and_reasoning():
    mod = _load_driver_module()
    driver = mod.Driver()
    messages = [{"role": "system", "content": "s"}]
    for i in range(60):
        messages.append({"role": "assistant", "content": "", "tool_calls": [], "reasoning_content": "R" * 20000})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "X" * 30000})
    driver._enforce_history_budget(messages)
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "[truncated]"
    assert tool_msgs[-1]["content"] != "[truncated]"  # recent context kept
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert "reasoning_content" not in assistants[0]
    assert "reasoning_content" in assistants[-1]


def test_context_length_error_is_not_retried():
    """A genuine context-length error must fail fast — retrying the identical
    oversized payload 4x would just reproduce the same error every time."""
    mod = _load_driver_module()
    driver = mod.Driver()

    calls = []

    class _Resp:
        status_code = 400
        text = '{"error": {"message": "This model'"'"'s maximum context length is 128000 tokens"}}'

        def json(self):
            return json.loads(self.text)

    class _Client:
        async def post(self, url, json=None):
            calls.append(1)
            return _Resp()

    try:
        asyncio.run(driver.llm(_Client(), [{"role": "user", "content": "hi"}]))
        raised = None
    except Exception as e:  # noqa: BLE001
        raised = e

    assert isinstance(raised, mod.ContextLengthExceededError)
    assert len(calls) == 1, "must not retry an unrecoverable context-length error"


def test_summarize_runs_harbor_qa_dance_and_rebuilds_history():
    """Harbor parity: _summarize is the three-subagent flow (summary ->
    questions from a FRESH context -> answers over full history), and the
    history is rebuilt to [system, question_prompt, questions] with the
    answers returned as the handoff."""
    mod = _load_driver_module()
    driver = mod.Driver()

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Task:\noriginal instruction"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call({"commands": []}, "c0")]},
        {"role": "tool", "tool_call_id": "c0", "content": "some output"},
    ]
    calls = []

    async def scripted_llm(client, msgs, use_tools=True, tool_choice=None):
        calls.append({"msgs": json.loads(json.dumps(msgs)), "use_tools": use_tools, "tool_choice": tool_choice})
        return {"content": ["THE-SUMMARY", "THE-QUESTIONS", "THE-ANSWERS"][len(calls) - 1]}

    driver.llm = scripted_llm
    handoff = asyncio.run(driver._summarize(None, messages, "original instruction", _FakeTmux()))

    # call 1: summary over full history, tools present but forced off
    assert calls[0]["tool_choice"] == "none"
    assert "You are about to hand off your work to another AI agent." in calls[0]["msgs"][-1]["content"]
    assert calls[0]["msgs"][0]["role"] == "system"
    assert any(m.get("role") == "tool" for m in calls[0]["msgs"])
    # call 2: questions agent starts FRESH — one user message, no tools
    assert calls[1]["use_tools"] is False
    assert len(calls[1]["msgs"]) == 1
    q_prompt = calls[1]["msgs"][0]["content"]
    assert "THE-SUMMARY" in q_prompt and "captured screen" in q_prompt and "original instruction" in q_prompt
    assert "Please begin by asking several questions" in q_prompt
    # call 3: answers agent sees full history + summary exchange + questions
    assert calls[2]["tool_choice"] == "none"
    assert calls[2]["msgs"][-1]["content"].startswith("The next agent has a few questions for you")
    assert "THE-QUESTIONS" in calls[2]["msgs"][-1]["content"]
    assert calls[2]["msgs"][-2] == {"role": "assistant", "content": "THE-SUMMARY"}
    # history rebuilt exactly like harbor's chat._messages replacement
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"] == q_prompt
    assert messages[2]["content"] == "THE-QUESTIONS"
    # handoff = answers wrapped in harbor's continuation text
    assert handoff.startswith("Here are the answers the other agent provided.\n\nTHE-ANSWERS")
    assert handoff.endswith("Please follow the spec to interact with the terminal.")


def test_recovery_falls_back_to_short_summary_when_dance_fails():
    """Tier 2: the full Q&A dance fails (its first call re-hits the context
    limit) -> a tiny no-history short-summary call produces the handoff,
    appended to the unwound history, then the chat is retried."""
    mod = _load_driver_module()
    driver = mod.Driver()

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Task:\noriginal instruction"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call({"commands": []}, "c0")]},
        {"role": "tool", "tool_call_id": "c0", "content": "some output"},
    ]
    calls = []

    async def scripted_llm(client, msgs, use_tools=True, tool_choice=None):
        calls.append({"msgs": json.loads(json.dumps(msgs)), "use_tools": use_tools})
        if len(calls) == 1:
            raise mod.ContextLengthExceededError("still too big")
        if len(calls) == 2:
            return {"content": "SHORT-SUMMARY"}
        return {"content": "", "tool_calls": [_tool_call({"commands": []}, "post")]}

    driver.llm = scripted_llm
    msg = asyncio.run(driver._recover_from_context_overflow(None, messages, "original instruction", _FakeTmux()))

    # tier-2 call carried NO history and harbor's short prompt
    assert len(calls[1]["msgs"]) == 1
    assert calls[1]["msgs"][0]["content"].startswith("Briefly continue this task: original instruction")
    assert calls[1]["use_tools"] is False
    # harbor's fallback-2 handoff format, appended to the (unwound) history
    assert messages[-1] == {"role": "user", "content": "original instruction\n\nSummary: SHORT-SUMMARY"}
    # and the retried chat's response is returned
    assert msg["tool_calls"][0]["id"] == "post"


def test_recovery_ultimate_fallback_and_technical_difficulties():
    """Tier 3 (no LLM) fires when both summary tiers fail; if even the
    retried chat fails, the driver fabricates harbor's 'Technical
    difficulties' reply instead of crashing the episode."""
    mod = _load_driver_module()
    driver = mod.Driver()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Task:\noriginal instruction"},
    ]

    async def always_fails(client, msgs, use_tools=True, tool_choice=None):
        raise RuntimeError("gateway down")

    driver.llm = always_fails
    msg = asyncio.run(driver._recover_from_context_overflow(None, messages, "original instruction", _FakeTmux()))

    assert messages[-1]["content"] == "original instruction\n\nCurrent state: captured screen"
    assert msg == {"content": "Technical difficulties. Please continue with the task."}


def test_context_length_error_mid_run_runs_dance_then_retries():
    """End-to-end through run(): main call hits the context limit -> Q&A
    dance -> retried main call sees [system, question_prompt, questions,
    handoff] and the episode continues to completion."""
    mod = _load_driver_module()
    tmux = _FakeTmux()
    mod.TmuxSession = lambda *a, **kw: tmux

    driver = mod.Driver()
    driver.max_steps = 5
    calls = []

    async def scripted_llm(client, messages, use_tools=True, tool_choice=None):
        calls.append(json.loads(json.dumps(messages)))
        n = len(calls)
        if n == 1:
            raise mod.ContextLengthExceededError("boom")
        if n == 2:
            return {"content": "SUM"}
        if n == 3:
            return {"content": "QS"}
        if n == 4:
            return {"content": "ANS"}
        return {"content": "", "tool_calls": [_tool_call({"commands": [], "task_complete": True}, f"tc{n}")]}

    driver.llm = scripted_llm
    asyncio.run(driver.run("solve the task"))

    assert len(calls) == 6  # main(CLE) + summary + questions + answers + retry + confirm
    retry = calls[4]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2]["content"] == "QS"
    assert retry[3]["content"].startswith("Here are the answers the other agent provided.\n\nANS")


def test_proactive_summarization_triggers_on_low_free_tokens():
    """Harbor parity: free tokens below the threshold at the top of a step ->
    the dance runs with no error anywhere; comfortable headroom -> no calls."""
    mod = _load_driver_module()
    driver = mod.Driver()
    driver.context_limit = 1000
    driver.proactive_threshold = 800  # trigger when estimate > 200 tokens

    calls = []

    async def scripted_llm(client, msgs, use_tools=True, tool_choice=None):
        calls.append(1)
        return {"content": ["SUM", "QS", "ANS"][len(calls) - 1]}

    driver.llm = scripted_llm

    small = [{"role": "system", "content": "s"}, {"role": "user", "content": "tiny"}]
    asyncio.run(driver._check_proactive_summarization(None, small, "task", _FakeTmux()))
    assert calls == [], "well under the limit: must not summarize"

    big = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task msg"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call({"commands": []}, "c0")]},
        {"role": "tool", "tool_call_id": "c0", "content": "X" * 4000},  # ~1000 estimated tokens
    ]
    asyncio.run(driver._check_proactive_summarization(None, big, "task", _FakeTmux()))
    assert len(calls) == 3, "low free tokens: full dance must run"
    assert [m["role"] for m in big] == ["system", "user", "assistant", "user"]
    assert big[-1]["content"].startswith("Here are the answers the other agent provided.")


def test_unwind_pops_whole_turns_from_the_tail():
    """Harbor unwinds by dropping the most recent messages; the native analog
    drops whole assistant+tool turns so no orphaned tool message survives."""
    mod = _load_driver_module()
    driver = mod.Driver()
    driver.context_limit = 1000  # tokens; chars/4 estimate

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(10):
        messages.append({"role": "assistant", "content": "", "tool_calls": [_tool_call({"commands": []}, f"c{i}")]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "X" * 2000})  # ~500 tokens/turn

    driver._unwind_messages_to_free_tokens(messages, target_free_tokens=400)

    est = driver._estimate_tokens(messages)
    assert driver.context_limit - est >= 400
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    # tail must end on a complete turn, never an orphaned assistant tool_call
    assert messages[-1]["role"] == "tool"
    kept_ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
    assert kept_ids == [f"c{i}" for i in range(len(kept_ids))], "must drop from the tail, keeping the oldest turns"


def test_summarize_off_propagates_context_length_error():
    """Harbor parity: enable_summarize=False means the error is raised (and
    recorded in the outcome), not silently truncated around."""
    mod = _load_driver_module()
    tmux = _FakeTmux()
    mod.TmuxSession = lambda *a, **kw: tmux

    driver = mod.Driver()
    driver.enable_summarize = False
    driver.max_steps = 3

    async def always_cle(client, messages, use_tools=True, tool_choice=None):
        raise mod.ContextLengthExceededError("boom")

    driver.llm = always_cle
    try:
        asyncio.run(driver.run("solve the task"))
        raised = None
    except Exception as e:  # noqa: BLE001
        raised = e
    assert isinstance(raised, mod.ContextLengthExceededError)


def test_llm_retries_heartbeat_wrapped_error_bodies():
    """The gateway heartbeat wraps late upstream failures as 200 + {'error': ...};
    the driver must treat those as retryable, not as a completion."""
    mod = _load_driver_module()
    driver = mod.Driver()

    class _Resp:
        def __init__(self, payload, status=200):
            self.status_code = status
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    responses = iter(
        [
            _Resp({"error": {"message": "gateway upstream failure", "type": "gateway_upstream_error"}}),
            _Resp({"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}),
        ]
    )

    class _Client:
        async def post(self, url, json=None):
            return next(responses)

    async def go():
        # collapse the retry backoff for the test
        sleeps = []
        real_sleep = asyncio.sleep
        mod.asyncio.sleep = lambda s: real_sleep(0) or sleeps.append(s) or real_sleep(0)
        try:
            return await driver.llm(_Client(), [{"role": "user", "content": "hi"}])
        finally:
            mod.asyncio.sleep = real_sleep

    msg = asyncio.run(go())
    assert msg["content"] == "ok"
    assert driver.in_tokens == 5


# ---------------------------------------------------------------------------
# Harness shell
# ---------------------------------------------------------------------------


def test_driver_script_is_valid_python():
    ast.parse(_NATIVE_DRIVER_SCRIPT)


def test_write_configs_ships_the_native_driver():
    from tests.harnesses.test_cli_harness import FakeSandbox, _make_config, _make_task

    h = Terminus2NativeHarness()
    sandbox = FakeSandbox()
    h.write_configs(sandbox, _make_task(), _make_config(), env={})
    driver_writes = [c.command for c in sandbox.calls if _DRIVER_PATH in c.command]
    assert driver_writes, "driver must be written to the standard driver path"
    assert "execute_commands" in driver_writes[0], "native tool schema must be in the shipped driver"
    assert "reasoning_content" in driver_writes[0]


def test_build_env_carries_summarization_knobs():
    from tests.harnesses.test_cli_harness import _make_config, _make_task

    h = Terminus2NativeHarness()
    env = h.build_env(_make_task(), _make_config())
    assert env["RLLM_TERMINUS_ENABLE_SUMMARIZE"] == "1"
    assert env["RLLM_TERMINUS_PROACTIVE_THRESHOLD"] == "8000"
    assert "RLLM_TERMINUS_CONTEXT_LIMIT" not in env, "unset limit must fall back to the driver default"

    h.context_limit_tokens = 240_000
    h.enable_summarize = False
    env = h.build_env(_make_task(), _make_config())
    assert env["RLLM_TERMINUS_CONTEXT_LIMIT"] == "240000"
    assert env["RLLM_TERMINUS_ENABLE_SUMMARIZE"] == "0"


def test_registry_resolves_native_harness():
    from rllm.eval.agent_loader import load_agent

    agent = load_agent("terminus2-native")
    assert type(agent).__name__ == "Terminus2NativeHarness"
    assert agent.name == "terminus2-native"
