"""Offline tests for the generic OpenHands harness."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType, SimpleNamespace

from rllm.eval.agent_loader import load_agent
from rllm.harnesses.openhands import (
    _DRIVER_SCRIPT,
    _OUTCOME_PATH,
    OPENHANDS_EXTENSIONS_REVISION,
    OPENHANDS_LIBTMUX_VERSION,
    OPENHANDS_LOCALE,
    OPENHANDS_SDK_REVISION,
    OpenHandsHarness,
)
from rllm.sandbox.protocol import SandboxCommandTimeout
from rllm.types import AgentConfig, Task, TerminationReason


@dataclass
class FakeSandbox:
    calls: list[str] = field(default_factory=list)
    outcome: str | None = None
    fail_invocation: bool = False
    timeout_invocation: bool = False

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        del timeout, user
        self.calls.append(command)
        if command.startswith(f"cat {_OUTCOME_PATH}"):
            return self.outcome or ""
        if "/opt/rllm/openhands/bin/python /tmp/rllm-openhands/driver.py" in command:
            if self.timeout_invocation:
                raise SandboxCommandTimeout("agent timed out")
            if self.fail_invocation:
                raise RuntimeError("driver exited non-zero")
        return "OK"

    def is_alive(self) -> bool:
        return True


def _task() -> Task:
    return Task(
        id="task-1",
        instruction="Fix the issue while keeping the API stable.",
        metadata={"workdir": "/repo", "agent_timeout": 3600.0},
    )


def _config(model: str = "vendor/model-v1") -> AgentConfig:
    return AgentConfig(
        base_url="http://gateway/sessions/test/v1",
        model=model,
        session_uid="test",
        metadata={"gateway_auth_token": "gateway-token"},
    )


def test_registry_loads_generic_openhands_harness():
    assert isinstance(load_agent("openhands"), OpenHandsHarness)


def test_install_script_pins_sdk_and_tmux_parser():
    script = OpenHandsHarness().install_script()

    assert OPENHANDS_SDK_REVISION in script
    assert "#subdirectory=openhands-sdk" in script
    assert "#subdirectory=openhands-tools" in script
    assert "--no-deps" in script
    assert f"libtmux=={OPENHANDS_LIBTMUX_VERSION}" in script
    assert f"{OPENHANDS_SDK_REVISION}:libtmux=={OPENHANDS_LIBTMUX_VERSION}" in script


def test_driver_is_valid_and_reads_agent_settings_from_environment():
    compile(_DRIVER_SCRIPT, "<openhands-driver>", "exec")

    assert 'os.environ["RLLM_OPENHANDS_TEMPERATURE"]' in _DRIVER_SCRIPT
    assert 'os.environ["RLLM_OPENHANDS_TOP_P"]' in _DRIVER_SCRIPT
    assert '_optional_int("RLLM_OPENHANDS_MAX_INPUT_TOKENS")' in _DRIVER_SCRIPT
    assert '_optional_int("RLLM_OPENHANDS_MAX_OUTPUT_TOKENS")' in _DRIVER_SCRIPT
    assert 'os.environ["RLLM_OPENHANDS_MAX_ITERATIONS"]' in _DRIVER_SCRIPT
    assert "get_default_tools(enable_browser=False)" in _DRIVER_SCRIPT
    assert "conversation.run()" in _DRIVER_SCRIPT
    assert "traceback.format_exc()" in _DRIVER_SCRIPT
    assert "SWE-bench" not in _DRIVER_SCRIPT
    assert "GLM" not in _DRIVER_SCRIPT


def test_build_env_routes_any_model_through_rllm_gateway():
    harness = OpenHandsHarness(
        temperature=0.2,
        top_p=0.9,
        max_input_tokens=123_000,
        max_output_tokens=4_096,
        max_iterations=17,
    )
    env = harness.build_env(_task(), _config("accounts/provider/models/model-v1"))

    assert env["LLM_MODEL"] == "openai/accounts/provider/models/model-v1"
    assert env["LLM_API_KEY"] == "gateway-token"
    assert env["LLM_BASE_URL"] == "http://gateway/sessions/test/v1"
    assert env["OPENAI_API_KEY"] == "gateway-token"
    assert env["EXTENSIONS_REF"] == OPENHANDS_EXTENSIONS_REVISION
    assert env["LC_ALL"] == OPENHANDS_LOCALE
    assert env["LANG"] == OPENHANDS_LOCALE
    assert env["RLLM_OPENHANDS_WORKDIR"] == "/repo"
    assert env["RLLM_OPENHANDS_TEMPERATURE"] == "0.2"
    assert env["RLLM_OPENHANDS_TOP_P"] == "0.9"
    assert env["RLLM_OPENHANDS_MAX_INPUT_TOKENS"] == "123000"
    assert env["RLLM_OPENHANDS_MAX_OUTPUT_TOKENS"] == "4096"
    assert env["RLLM_OPENHANDS_MAX_ITERATIONS"] == "17"


def test_build_env_does_not_copy_a_host_provider_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "host-provider-secret")
    config = _config()
    config.metadata = {}

    env = OpenHandsHarness().build_env(_task(), config)

    assert env["LLM_API_KEY"] == "sk-rllm-gateway"
    assert env["OPENAI_API_KEY"] == "sk-rllm-gateway"


def test_write_configs_uses_raw_task_instruction_by_default():
    harness = OpenHandsHarness()
    sandbox = FakeSandbox()
    task = _task()
    config = _config()
    env = harness.build_env(task, config)

    harness.write_configs(sandbox, task, config, env)

    assert len(sandbox.calls) == 2
    assert "Local-workspace OpenHands driver" in sandbox.calls[0]
    assert task.instruction in sandbox.calls[1]
    assert "<issue_description>" not in sandbox.calls[1]


def _run_driver_with_fake_openhands(monkeypatch, tmp_path, *, finish_before_error: bool) -> dict:
    class ExecutionStatus(Enum):
        FINISHED = "finished"

    class FinishAction:
        pass

    class ActionEvent:
        def __init__(self, action):
            self.action = action

    class MessageEvent:
        def __init__(self, source):
            self.source = source

    class LLM:
        def __init__(self, **kwargs):
            del kwargs

        def model_copy(self, **kwargs):
            del kwargs
            return self

    class AcceptsKeywords:
        def __init__(self, **kwargs):
            del kwargs

    class Conversation:
        def __init__(self, **kwargs):
            del kwargs
            self.state = SimpleNamespace(events=[], execution_status=ExecutionStatus.FINISHED)

        def send_message(self, message):
            del message

        def run(self):
            if finish_before_error:
                self.state.events.append(ActionEvent(FinishAction()))
                raise RuntimeError("cleanup failed after finish")
            raise RuntimeError("gateway unavailable")

    modules = {
        name: ModuleType(name)
        for name in (
            "openhands",
            "openhands.sdk",
            "openhands.sdk.context",
            "openhands.sdk.context.condenser",
            "openhands.sdk.conversation",
            "openhands.sdk.conversation.state",
            "openhands.sdk.event",
            "openhands.sdk.skills",
            "openhands.sdk.tool",
            "openhands.sdk.tool.builtins",
            "openhands.sdk.tool.builtins.finish",
            "openhands.tools",
            "openhands.tools.preset",
            "openhands.tools.preset.default",
        )
    }
    modules["openhands.sdk"].Agent = AcceptsKeywords
    modules["openhands.sdk"].Conversation = Conversation
    modules["openhands.sdk"].LLM = LLM
    modules["openhands.sdk.context"].AgentContext = AcceptsKeywords
    modules["openhands.sdk.context.condenser"].LLMSummarizingCondenser = AcceptsKeywords
    modules["openhands.sdk.conversation.state"].ConversationExecutionStatus = ExecutionStatus
    modules["openhands.sdk.event"].ActionEvent = ActionEvent
    modules["openhands.sdk.event"].MessageEvent = MessageEvent
    modules["openhands.sdk.skills"].load_public_skills = lambda: []
    modules["openhands.sdk.tool.builtins.finish"].FinishAction = FinishAction
    modules["openhands.tools.preset.default"].get_default_tools = lambda **kwargs: []
    for name, module in modules.items():
        if name.rsplit(".", 1)[-1] not in {"condenser", "state", "event", "finish", "skills", "default"}:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    prompt_path = tmp_path / "prompt.txt"
    summary_path = tmp_path / "summary.json"
    outcome_path = tmp_path / "outcome.json"
    prompt_path.write_text("Fix it")
    env = {
        "LLM_MODEL": "openai/model-v1",
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "http://gateway/v1",
        "RLLM_OPENHANDS_WORKDIR": "/app",
        "RLLM_OPENHANDS_PROMPT_PATH": str(prompt_path),
        "RLLM_OPENHANDS_SUMMARY_PATH": str(summary_path),
        "RLLM_OPENHANDS_OUTCOME_PATH": str(outcome_path),
        "RLLM_OPENHANDS_TEMPERATURE": "1.0",
        "RLLM_OPENHANDS_TOP_P": "1.0",
        "RLLM_OPENHANDS_MAX_ITERATIONS": "500",
        "RLLM_OPENHANDS_CONDENSER_MAX_SIZE": "240",
        "RLLM_OPENHANDS_CONDENSER_KEEP_FIRST": "2",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    namespace = {"__name__": "test_openhands_driver"}
    exec(compile(_DRIVER_SCRIPT, "<openhands-driver>", "exec"), namespace)
    namespace["main"]()
    return json.loads(outcome_path.read_text())


def test_driver_treats_typed_finish_as_authoritative(monkeypatch, tmp_path):
    outcome = _run_driver_with_fake_openhands(monkeypatch, tmp_path, finish_before_error=True)

    assert outcome == {
        "execution_status": "finished",
        "exception_type": None,
        "finished": True,
        "message": "cleanup failed after finish",
        "traceback": None,
    }


def test_driver_records_failure_and_full_traceback_before_finish(monkeypatch, tmp_path):
    outcome = _run_driver_with_fake_openhands(monkeypatch, tmp_path, finish_before_error=False)

    assert outcome["finished"] is False
    assert outcome["exception_type"] == "RuntimeError"
    assert outcome["message"] == "gateway unavailable"
    assert "RuntimeError: gateway unavailable" in outcome["traceback"]


def test_run_maps_structured_outcomes():
    success = FakeSandbox(outcome=json.dumps({"finished": True}))
    failure = FakeSandbox(
        outcome=json.dumps(
            {
                "finished": False,
                "exception_type": "RuntimeError",
                "message": "gateway unavailable",
                "traceback": "RuntimeError: gateway unavailable",
            }
        )
    )

    success_episode = OpenHandsHarness().run(_task(), _config(), env=success)
    failure_episode = OpenHandsHarness().run(_task(), _config(), env=failure)

    assert success_episode.termination_reason is None
    assert failure_episode.termination_reason == TerminationReason.ERROR
    assert failure_episode.metadata["error"] == {
        "error_type": "RuntimeError",
        "message": "gateway unavailable\n\nRuntimeError: gateway unavailable",
    }


def test_run_maps_structured_agent_timeout():
    sandbox = FakeSandbox(
        outcome=json.dumps(
            {
                "finished": False,
                "exception_type": "AgentTimeoutError",
                "message": "agent timed out",
            }
        ),
        timeout_invocation=True,
    )

    episode = OpenHandsHarness().run(_task(), _config(), env=sandbox)

    assert episode.termination_reason == TerminationReason.TIMEOUT
