from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import httpx
from openai import APITimeoutError, OpenAI

from rllm.harnesses.native_react import (
    NATIVE_SYSTEM_PROMPT,
    NATIVE_TOOL_SCHEMAS,
    NativeReactHarness,
    PersistentBashSession,
    _is_timeout_exception,
    initial_messages,
    limit_output_length,
    parse_native_tool_calls,
    preserve_assistant_message,
    tool_result,
)
from rllm.sandbox.backends.local import LocalSandbox
from rllm.sandbox.protocol import SandboxCommandTimeout
from rllm.types import AgentConfig, Task, TerminationReason


def test_initial_prompt_and_native_tool_result_shape():
    messages = initial_messages("fix it", "/app\nfile.txt")

    assert (
        NATIVE_SYSTEM_PROMPT
        == """You are an expert software engineer solving a task inside a Linux container.

You have two tools available:
- **bash**: Execute a shell command and see the output. Use this to explore, edit files, compile, run programs, etc.
- **submit**: Mark the task as complete. Only call this when you are confident the task is fully solved.

Guidelines:
- Read the task description carefully before starting.
- Explore the environment first (ls, cat, pwd) to understand the setup.
- Work step by step. Run one command at a time and check the output before proceeding.
- If a command fails, analyze the error and try a different approach.
- When editing files, use heredocs (cat > file << 'EOF'), sed, or echo/printf. There is no interactive editor.
- Test your solution before submitting.
- When done, call the submit tool."""
    )
    assert messages == [
        {"role": "system", "content": NATIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Task:\nfix it\n\nCurrent directory:\n/app\nfile.txt",
        },
    ]
    assert tool_result("call-7", "ok") == {
        "role": "tool",
        "tool_call_id": "call-7",
        "content": "ok",
    }


def test_assistant_message_round_trips_all_fields():
    original = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "inspect first",
        "tool_calls": [
            {
                "id": "call-7",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
            }
        ],
        "provider_extension": {"future": [1, 2, 3]},
    }

    preserved = preserve_assistant_message(original)

    assert preserved == original
    assert preserved is not original


def test_assistant_message_adds_reasoning_content_alias_without_dropping_reasoning():
    original = {"role": "assistant", "content": "", "reasoning": "legacy field", "vendor": True}

    preserved = preserve_assistant_message(original)

    assert preserved["reasoning"] == "legacy field"
    assert preserved["reasoning_content"] == "legacy field"
    assert preserved["vendor"] is True


def test_assistant_sdk_message_keeps_explicit_null_and_extra_fields():
    class SDKMessage:
        def model_dump(self, **kwargs):
            assert kwargs == {"exclude_unset": True}
            return {
                "role": "assistant",
                "content": None,
                "reasoning_content": "think",
                "tool_calls": [],
                "provider_extension": {"keep": None},
            }

    assert preserve_assistant_message(SDKMessage()) == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "think",
        "tool_calls": [],
        "provider_extension": {"keep": None},
    }


def test_openai_client_sends_preserved_fields_without_filtering():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    client = OpenAI(
        base_url="http://gateway.test/v1",
        api_key="EMPTY",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "think",
        "tool_calls": [],
        "provider_extension": {"keep": None},
    }
    try:
        client.chat.completions.create(model="test", messages=[assistant])
    finally:
        client.close()

    assert captured["messages"] == [assistant]


def test_native_tool_calls_preserve_ids_order_and_arguments():
    tool_calls = parse_native_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call-bash",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                },
                {
                    "id": "call-submit",
                    "function": {"name": "submit", "arguments": "{}"},
                },
            ]
        }
    )

    assert [(call.id, call.name, call.arguments) for call in tool_calls] == [
        ("call-bash", "bash", {"command": "pwd"}),
        ("call-submit", "submit", {}),
    ]


def test_output_truncation_preserves_head_and_tail():
    text = "a" * 10 + "b" * 10
    assert limit_output_length(text, 10) == "aaaaa\n\n[... 10 characters elided ...]\n\nbbbbb"


def test_rollout_timeout_env_is_a_hard_cap(monkeypatch):
    monkeypatch.setenv("RLLM_HARNESS_RUN_TIMEOUT_S", "30")
    harness = NativeReactHarness()

    assert harness._effective_timeout(Task(id="long", instruction="x", metadata={"agent_timeout": 90})) == 30
    assert harness._effective_timeout(Task(id="short", instruction="x", metadata={"agent_timeout": 10})) == 10
    assert harness._effective_timeout(Task(id="unset", instruction="x", metadata={})) == 30


def test_timeout_exception_detection_covers_backend_and_sandbox_timeouts():
    class BackendTimeoutError(RuntimeError):
        status_code = 504

    assert _is_timeout_exception(BackendTimeoutError("gateway deadline exceeded"))
    assert _is_timeout_exception(SandboxCommandTimeout("command timed out"))
    assert not _is_timeout_exception(RuntimeError("ordinary failure"))


def test_persistent_bash_preserves_cwd_environment_and_functions():
    sandbox = LocalSandbox("native-react-persistence")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        first = shell.run(
            "cd /tmp; export NATIVE_REACT_STATE=kept; function state_probe(){ printf function-ok; }; pwd",
            timeout=10,
        )
        second = shell.run('pwd; printf "%s\\n" "$NATIVE_REACT_STATE"; state_probe', timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert first.output == "/tmp\n"
    assert second.output == "/tmp\nkept\nfunction-ok"


def test_harness_resends_complete_assistant_and_native_tool_result(monkeypatch):
    import openai

    requests = []
    first_message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I should inspect with bash.",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"printf command-ok"}',
                },
            }
        ],
        "provider_extension": {"keep": None},
    }

    class SDKMessage:
        def __init__(self, data):
            self.data = data

        def model_dump(self, **kwargs):
            assert kwargs == {"exclude_unset": True}
            return copy.deepcopy(self.data)

    class Completions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            if len(requests) == 1:
                message = first_message
            else:
                message = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "The command succeeded.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "submit", "arguments": "{}"},
                        }
                    ],
                }
            return SimpleNamespace(choices=[SimpleNamespace(message=SDKMessage(message))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs == {
                "base_url": "http://gateway/v1",
                "api_key": "EMPTY",
                "max_retries": 0,
            }
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-loop")
    try:
        episode = NativeReactHarness(max_turns=4).run(
            Task(
                id="task-1",
                instruction="inspect",
                metadata={"workdir": "/tmp", "agent_timeout": 30},
            ),
            AgentConfig(
                base_url="http://gateway/v1",
                model="qwen",
                session_uid="session-1",
            ),
            env=sandbox,
        )
    finally:
        sandbox.close()

    assert len(requests) == 2
    assert requests[0]["tools"] == NATIVE_TOOL_SCHEMAS
    assert requests[1]["tools"] == NATIVE_TOOL_SCHEMAS
    assert requests[1]["messages"][-2] == first_message
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "command-ok",
    }
    assert episode.task == "task-1"
    assert episode.metadata["native_react"] == {"turns": 2, "parse_errors": 0}


def test_api_timeout_returns_typed_timeout_episode(monkeypatch):
    import openai

    class Completions:
        def create(self, **kwargs):
            raise APITimeoutError(httpx.Request("POST", "http://gateway/v1/chat/completions"))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs == {
                "base_url": "http://gateway/v1",
                "api_key": "EMPTY",
                "max_retries": 0,
            }
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-api-timeout")
    try:
        episode = NativeReactHarness(max_turns=1).run(
            Task(
                id="timeout-task",
                instruction="inspect",
                metadata={"workdir": "/tmp", "agent_timeout": 30},
            ),
            AgentConfig(
                base_url="http://gateway/v1",
                model="qwen",
                session_uid="timeout-session",
            ),
            env=sandbox,
        )
    finally:
        sandbox.close()

    assert episode.termination_reason is TerminationReason.TIMEOUT
    assert episode.metadata["error"]["error_type"] == "AgentTimeoutError"
    assert "APITimeoutError" in episode.metadata["error"]["message"]
