from __future__ import annotations

import json

import pytest

from rllm.eval.agent_loader import load_agent
from rllm.harnesses.stirrup import _DRIVER_SCRIPT, StirrupHarness, _usage_metrics
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
    assert "aggregate_metadata" in _DRIVER_SCRIPT
    assert 'Path("/tmp/stirrup/run.json")' in _DRIVER_SCRIPT


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


def test_usage_metrics_include_tokens_and_priced_cost(tmp_path, monkeypatch):
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps(
            {
                "models": {
                    "z-ai/glm-5.2": {
                        "input": 1.4,
                        "answer": 4.4,
                        "reasoning": 4.4,
                    }
                }
            }
        )
    )
    monkeypatch.setenv("RLLM_PRICING_FILE", str(pricing))

    metrics = _usage_metrics(
        {"metadata": {"token_usage": [{"input": 1_000_000, "answer": 100_000, "reasoning": 50_000}]}},
        "z-ai/glm-5.2",
    )

    assert metrics == {
        "input_tokens": 1_000_000,
        "answer_tokens": 100_000,
        "reasoning_tokens": 50_000,
        "output_tokens": 150_000,
        "total_tokens": 1_150_000,
        "cost_usd": pytest.approx(2.06),
    }


def test_stirrup_run_persists_deliverables_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))

    class FakeSandbox:
        def exec(self, command, timeout=None, user=None):
            del timeout, user
            if command == "cat /tmp/stirrup/run.json":
                return json.dumps(
                    {
                        "finish": {"paths": ["Sample.xlsx"]},
                        "metadata": {"token_usage": [{"input": 10, "answer": 3, "reasoning": 2}]},
                    }
                )
            return ""

        def download_dir(self, remote_path, local_path):
            assert remote_path == "/workspace/deliverables"
            output = tmp_path / "rllm-home" / "agent_outputs" / "test" / "Sample.xlsx"
            assert str(output.parent) == local_path
            output.parent.mkdir(parents=True)
            output.write_bytes(b"xlsx")
            return [str(output)]

    episode = StirrupHarness().run(_task(), _config(), env=FakeSandbox())

    assert episode.trajectories[0].name == "stirrup"
    assert episode.metrics["total_tokens"] == 15
    assert episode.artifacts["submitted_paths"] == ["Sample.xlsx"]
    assert episode.artifacts["deliverables"] == [str(tmp_path / "rllm-home" / "agent_outputs" / "test" / "Sample.xlsx")]
