"""Offline tests for the SWE-bench Pro OpenHands GLM-5.2 profile."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rllm.eval.agent_loader import load_agent
from rllm.harnesses.swebench_pro_openhands_glm52 import (
    _DRIVER_SCRIPT,
    CONTEXT_WINDOW_TOKENS,
    LLM_STATS_TARGET_SCORE,
    MAX_ITERATIONS,
    MAX_OUTPUT_TOKENS,
    OPENHANDS_BENCHMARKS_REVISION,
    OPENHANDS_EXTENSIONS_REVISION,
    OPENHANDS_SDK_REVISION,
    REPRODUCTION_PROFILE,
    SwebenchProOpenHandsGLM52Harness,
    _render_prompt,
)
from rllm.types import AgentConfig, Task


@dataclass
class FakeSandbox:
    calls: list[str] = field(default_factory=list)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        del timeout, user
        self.calls.append(command)
        return "OK"


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
    assert REPRODUCTION_PROFILE["status"] == "best_effort"
    assert REPRODUCTION_PROFILE["score_status"] == "self_reported_unverified"
    assert "undisclosed" in REPRODUCTION_PROFILE["prompt_provenance"]
    assert REPRODUCTION_PROFILE["max_iterations"] == 500


def test_install_script_pins_sdk_and_uses_minimal_tool_dependencies():
    script = SwebenchProOpenHandsGLM52Harness().install_script()

    assert OPENHANDS_SDK_REVISION in script
    assert "#subdirectory=openhands-sdk" in script
    assert "#subdirectory=openhands-tools" in script
    assert "--no-deps" in script
    assert "binaryornot cachetools libtmux func-timeout" in script
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


def test_build_env_routes_openhands_through_the_rllm_gateway():
    env = SwebenchProOpenHandsGLM52Harness().build_env(_task(), _config())

    assert env["LLM_MODEL"] == "openai/glm-5.2"
    assert env["LLM_API_KEY"] == "gateway-token"
    assert env["LLM_BASE_URL"] == "http://gateway/sessions/test/v1"
    assert env["OPENAI_API_KEY"] == "gateway-token"
    assert env["EXTENSIONS_REF"] == OPENHANDS_EXTENSIONS_REVISION
    assert env["CONVERSATION_TIMEOUT"] == "3600"
    assert env["RLLM_OPENHANDS_WORKDIR"] == "/app"


def test_profile_rejects_non_glm52_models():
    with pytest.raises(ValueError, match="requires a GLM-5.2 model"):
        SwebenchProOpenHandsGLM52Harness().build_env(_task(), _config("openai/gpt-5.4"))


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


def test_registry_loads_profile_and_episode_records_provenance():
    harness = load_agent("swebench-pro-openhands-glm52")
    episode = harness._outcome_episode(_task())

    assert isinstance(harness, SwebenchProOpenHandsGLM52Harness)
    assert episode.metadata["reproduction_profile"] == REPRODUCTION_PROFILE
