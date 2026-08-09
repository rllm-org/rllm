from __future__ import annotations

import json

import pytest

from rllm.eval.agent_loader import load_agent
from rllm.harnesses.stirrup import _DRIVER_SCRIPT, StirrupHarness
from rllm.types import AgentConfig, Task


def _task() -> Task:
    return Task(
        id="gdpval-1",
        instruction="Read source.xlsx and create report.docx.",
        metadata={"workdir": "/workspace", "reference_files": ["source.xlsx"]},
    )


def _config(**sampling_params) -> AgentConfig:
    return AgentConfig(
        base_url="http://gateway/sessions/test/v1",
        model="z-ai/glm-5.2",
        session_uid="test",
        sampling_params=sampling_params,
        metadata={"gateway_auth_token": "gateway-token"},
    )


def test_stirrup_is_registered_as_a_builtin_agent():
    assert isinstance(load_agent("stirrup"), StirrupHarness)


def test_stirrup_uses_an_isolated_python_and_local_code_backend():
    harness = StirrupHarness()

    assert "uv venv --python 3.12" in harness.install_script()
    assert "stirrup==0.1.12" in harness.install_script()
    assert "LocalCodeExecToolProvider" in _DRIVER_SCRIPT
    assert "DockerCodeExecToolProvider" not in _DRIVER_SCRIPT


def test_stirrup_env_routes_model_through_gateway(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-secret")
    env = StirrupHarness().build_env(_task(), _config(reasoning_effort="xhigh"))

    assert env["OPENAI_BASE_URL"] == "http://gateway/sessions/test/v1"
    assert env["OPENAI_API_KEY"] == "gateway-token"
    assert env["RLLM_STIRRUP_MODEL"] == "z-ai/glm-5.2"
    assert env["RLLM_STIRRUP_REASONING_EFFORT"] == "xhigh"
    assert json.loads(env["RLLM_STIRRUP_REFERENCE_FILES"]) == ["source.xlsx"]
    assert env["BRAVE_API_KEY"] == "brave-secret"


def test_reasoning_effort_must_be_a_string():
    with pytest.raises(ValueError, match="must be a string"):
        StirrupHarness().build_env(_task(), _config(reasoning_effort=3))


def test_invocation_runs_stirrup_in_the_existing_sandbox():
    command = StirrupHarness().build_invocation("task", _task(), _config())

    assert command.startswith("/opt/stirrup-venv/bin/python /opt/stirrup/driver.py")
    assert "docker" not in command.lower()
