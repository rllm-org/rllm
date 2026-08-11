"""Offline tests for the SWE-bench Pro OpenHands GLM-5.2 profile."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from rllm.eval.agent_loader import load_agent
from rllm.harnesses.swebench_pro_openhands_glm52 import (
    _DRIVER_SCRIPT,
    _OUTCOME_PATH,
    CONTEXT_WINDOW_TOKENS,
    FIREWORKS_MODEL_ID,
    LLM_STATS_TARGET_SCORE,
    MAX_ITERATIONS,
    MAX_OUTPUT_TOKENS,
    OPENHANDS_BENCHMARKS_REVISION,
    OPENHANDS_EXTENSIONS_REVISION,
    OPENHANDS_LIBTMUX_VERSION,
    OPENHANDS_LOCALE,
    OPENHANDS_SDK_REVISION,
    OPENROUTER_MODEL_ID,
    OPENROUTER_PROVIDER_SLUG,
    REPRODUCTION_PROFILE,
    SwebenchProOpenHandsGLM52Harness,
    _render_prompt,
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
        if "/bin/python /tmp/rllm-swebench-pro-openhands.py" in command:
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
        instruction="Fix the issue.\n\nRequirements:\nKeep the API stable.",
        metadata={
            "workdir": "/app",
            "agent_timeout": 3600.0,
            "metadata": {"base_commit": "a" * 40},
        },
    )


def _config(model: str = "openai/glm-5.2") -> AgentConfig:
    return AgentConfig(
        base_url="http://gateway/sessions/test/v1",
        model=model,
        session_uid="test",
        metadata={"gateway_auth_token": "gateway-token"},
    )


def test_profile_records_public_baseline_and_disclosure_gap():
    assert LLM_STATS_TARGET_SCORE == 0.621
    assert len(OPENHANDS_BENCHMARKS_REVISION) == 40
    assert len(OPENHANDS_SDK_REVISION) == 40
    assert len(OPENHANDS_EXTENSIONS_REVISION) == 40
    assert OPENHANDS_LIBTMUX_VERSION == "0.53.0"
    assert OPENHANDS_LOCALE == "C.UTF-8"
    assert REPRODUCTION_PROFILE["status"] == "best_effort"
    assert REPRODUCTION_PROFILE["score_status"] == "self_reported_unverified"
    assert "undisclosed" in REPRODUCTION_PROFILE["prompt_provenance"]
    assert REPRODUCTION_PROFILE["max_iterations"] == 500
    assert REPRODUCTION_PROFILE["openrouter_provider_route"] == {
        "only": [OPENROUTER_PROVIDER_SLUG],
        "allow_fallbacks": False,
        "require_parameters": False,
    }


def test_install_script_pins_sdk_and_uses_minimal_tool_dependencies():
    script = SwebenchProOpenHandsGLM52Harness().install_script()

    assert OPENHANDS_SDK_REVISION in script
    assert "#subdirectory=openhands-sdk" in script
    assert "#subdirectory=openhands-tools" in script
    assert "--no-deps" in script
    assert f"binaryornot cachetools libtmux=={OPENHANDS_LIBTMUX_VERSION} func-timeout" in script
    assert f"{OPENHANDS_SDK_REVISION}:libtmux=={OPENHANDS_LIBTMUX_VERSION}" in script
    assert "tmux" in script


def test_driver_is_valid_and_encodes_disclosed_sampling_settings():
    compile(_DRIVER_SCRIPT, "<openhands-driver>", "exec")

    assert f"temperature={1.0}" in _DRIVER_SCRIPT
    assert f"top_p={1.0}" in _DRIVER_SCRIPT
    assert f"max_input_tokens={CONTEXT_WINDOW_TOKENS}" in _DRIVER_SCRIPT
    assert f"max_output_tokens={MAX_OUTPUT_TOKENS}" in _DRIVER_SCRIPT
    assert f"max_iteration_per_run={MAX_ITERATIONS}" in _DRIVER_SCRIPT
    assert "get_default_tools(enable_browser=False)" in _DRIVER_SCRIPT
    assert "LLMSummarizingCondenser" in _DRIVER_SCRIPT
    assert "load_public_skills()" in _DRIVER_SCRIPT
    assert "agent_context=agent_context" in _DRIVER_SCRIPT
    assert 'model.startswith("openrouter/")' in _DRIVER_SCRIPT
    assert '"allow_fallbacks": False' in _DRIVER_SCRIPT
    assert '"require_parameters": False' in _DRIVER_SCRIPT
    assert "conversation.run()" in _DRIVER_SCRIPT
    assert "conversation.run(timeout=" not in _DRIVER_SCRIPT
    assert "finished = _finished(events)" in _DRIVER_SCRIPT
    assert '"exception_type": None if finished else type(failure).__name__' in _DRIVER_SCRIPT
    assert "failure_traceback = traceback.format_exc()" in _DRIVER_SCRIPT
    assert '"traceback": None if finished else failure_traceback' in _DRIVER_SCRIPT
    assert "os.replace(temporary, path)" in _DRIVER_SCRIPT


def test_build_env_routes_openhands_through_the_rllm_gateway():
    env = SwebenchProOpenHandsGLM52Harness().build_env(_task(), _config())

    assert env["LLM_MODEL"] == "openai/glm-5.2"
    assert env["LLM_API_KEY"] == "gateway-token"
    assert env["LLM_BASE_URL"] == "http://gateway/sessions/test/v1"
    assert env["OPENAI_API_KEY"] == "gateway-token"
    assert env["EXTENSIONS_REF"] == OPENHANDS_EXTENSIONS_REVISION
    assert env["LC_ALL"] == OPENHANDS_LOCALE
    assert env["LANG"] == OPENHANDS_LOCALE
    assert "CONVERSATION_TIMEOUT" not in env
    assert env["RLLM_OPENHANDS_WORKDIR"] == "/app"
    assert env["RLLM_OPENHANDS_OUTCOME_PATH"] == _OUTCOME_PATH


def test_profile_rejects_non_glm52_models():
    with pytest.raises(ValueError, match="requires a GLM-5.2 model"):
        SwebenchProOpenHandsGLM52Harness().build_env(_task(), _config("openai/gpt-5.4"))


def test_build_env_preserves_openrouter_model_alias():
    env = SwebenchProOpenHandsGLM52Harness().build_env(_task(), _config(OPENROUTER_MODEL_ID))

    assert env["LLM_MODEL"] == f"openrouter/{OPENROUTER_MODEL_ID}"


def test_build_env_preserves_fireworks_model_alias_through_gateway():
    env = SwebenchProOpenHandsGLM52Harness().build_env(_task(), _config(FIREWORKS_MODEL_ID))

    assert env["LLM_MODEL"] == f"openai/{FIREWORKS_MODEL_ID}"


def test_build_env_does_not_fall_back_to_a_host_provider_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "host-provider-secret")
    config = _config()
    config.metadata = {}

    env = SwebenchProOpenHandsGLM52Harness().build_env(_task(), config)

    assert env["LLM_API_KEY"] == "sk-rllm-gateway"
    assert env["OPENAI_API_KEY"] == "sk-rllm-gateway"


def test_write_configs_materializes_driver_and_public_prompt():
    harness = SwebenchProOpenHandsGLM52Harness()
    sandbox = FakeSandbox()
    task = _task()
    config = _config()
    env = harness.build_env(task, config)

    harness.write_configs(sandbox, task, config, env)

    assert len(sandbox.calls) == 2
    assert "Pinned local-workspace OpenHands driver" in sandbox.calls[0]
    assert "<issue_description>" in sandbox.calls[1]
    assert "Keep the API stable." in sandbox.calls[1]
    assert "a" * 40 in sandbox.calls[1]
    assert "make the minimal changes to non-test files in the /app directory" in sandbox.calls[1]


def test_prompt_and_invocation_use_the_existing_rllm_task_workspace():
    prompt = _render_prompt("Fix it", "/app", "b" * 40)
    invocation = SwebenchProOpenHandsGLM52Harness().build_invocation("Fix it", _task(), _config())

    assert "code repository in the directory /app" in prompt
    assert "b" * 40 in prompt
    assert invocation.startswith("set -o pipefail;")
    assert "/opt/rllm/swebench-pro-openhands-glm52/bin/python" in invocation
    assert "tee /tmp/swebench-pro-openhands-glm52.log" in invocation
    assert "grep" not in invocation


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
            "openhands.sdk.tool",
            "openhands.sdk.tool.builtins",
            "openhands.sdk.tool.builtins.finish",
            "openhands.sdk.skills",
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
    modules["openhands.sdk.tool.builtins.finish"].FinishAction = FinishAction
    modules["openhands.sdk.skills"].load_public_skills = lambda: []
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
        "LLM_MODEL": "openrouter/z-ai/glm-5.2",
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "http://gateway/v1",
        "RLLM_OPENHANDS_WORKDIR": "/app",
        "RLLM_OPENHANDS_PROMPT_PATH": str(prompt_path),
        "RLLM_OPENHANDS_SUMMARY_PATH": str(summary_path),
        "RLLM_OPENHANDS_OUTCOME_PATH": str(outcome_path),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    namespace = {"__name__": "test_openhands_driver"}
    exec(compile(_DRIVER_SCRIPT, "<openhands-driver>", "exec"), namespace)
    namespace["main"]()
    return json.loads(outcome_path.read_text())


def test_driver_treats_typed_finish_as_authoritative_after_cleanup_error(monkeypatch, tmp_path):
    outcome = _run_driver_with_fake_openhands(monkeypatch, tmp_path, finish_before_error=True)

    assert outcome == {
        "execution_status": "finished",
        "exception_type": None,
        "finished": True,
        "message": "cleanup failed after finish",
        "traceback": None,
    }


def test_driver_records_gateway_failure_when_no_typed_finish_exists(monkeypatch, tmp_path):
    outcome = _run_driver_with_fake_openhands(monkeypatch, tmp_path, finish_before_error=False)

    assert {key: value for key, value in outcome.items() if key != "traceback"} == {
        "execution_status": "finished",
        "exception_type": "RuntimeError",
        "finished": False,
        "message": "gateway unavailable",
    }
    assert "RuntimeError: gateway unavailable" in outcome["traceback"]


def test_read_outcome_uses_typed_finish_not_reward_or_stdout():
    sandbox = FakeSandbox(outcome=json.dumps({"finished": True, "reward": 0, "message": "cleanup failed"}))

    assert SwebenchProOpenHandsGLM52Harness()._read_outcome(sandbox) == {}


def test_run_maps_typed_finish_to_env_done():
    sandbox = FakeSandbox(
        outcome=json.dumps(
            {
                "finished": True,
                "execution_status": "finished",
                "exception_type": None,
                "message": "cleanup failed after finish",
            }
        )
    )

    episode = SwebenchProOpenHandsGLM52Harness().run(_task(), _config(), env=sandbox)

    assert episode.termination_reason is None
    assert "error" not in episode.metadata


def test_run_maps_structured_failure_without_finish_to_error():
    sandbox = FakeSandbox(
        outcome=json.dumps(
            {
                "finished": False,
                "execution_status": "error",
                "exception_type": "RuntimeError",
                "message": "gateway unavailable",
                "traceback": "Traceback (most recent call last):\n  driver.py, line 1\nRuntimeError: gateway unavailable\n",
            }
        )
    )

    episode = SwebenchProOpenHandsGLM52Harness().run(_task(), _config(), env=sandbox)

    assert episode.termination_reason == TerminationReason.ERROR
    assert episode.metadata["error"] == {
        "error_type": "RuntimeError",
        "message": "gateway unavailable\n\nTraceback (most recent call last):\n  driver.py, line 1\nRuntimeError: gateway unavailable\n",
    }


def test_run_maps_structured_timeout_without_finish_to_timeout():
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

    episode = SwebenchProOpenHandsGLM52Harness().run(_task(), _config(), env=sandbox)

    assert episode.termination_reason == TerminationReason.TIMEOUT


def test_run_keeps_nonzero_without_sentinel_as_error():
    sandbox = FakeSandbox(fail_invocation=True)

    episode = SwebenchProOpenHandsGLM52Harness().run(_task(), _config(), env=sandbox)

    assert episode.termination_reason == TerminationReason.ERROR


def test_registry_loads_profile_and_episode_records_provenance():
    harness = load_agent("swebench-pro-openhands-glm52")
    episode = harness._outcome_episode(_task())

    assert isinstance(harness, SwebenchProOpenHandsGLM52Harness)
    assert episode.metadata["reproduction_profile"] == REPRODUCTION_PROFILE
