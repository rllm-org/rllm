"""Verify one timeout override reaches housekeeping calls across the pipeline."""

from unittest.mock import Mock

import pytest

from rllm.eval._resolution import _setup_task_environment
from rllm.eval.script_evaluator import ShellScriptEvaluator
from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness
from rllm.sandbox.timeouts import sandbox_control_timeout_s
from rllm.types import Episode, Task


@pytest.mark.parametrize(("override", "expected"), [(None, 10), ("120", 120)])
def test_housekeeping_timeout_reaches_setup_grading_and_outcome(monkeypatch, tmp_path, override, expected):
    monkeypatch.delenv("RLLM_SANDBOX_CONTROL_TIMEOUT_S", raising=False)
    if override is not None:
        monkeypatch.setenv("RLLM_SANDBOX_CONTROL_TIMEOUT_S", override)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    task = Task(id="task", instruction="", dataset_dir=tmp_path, metadata={"agent_user": "agent"})
    sandbox = Mock()

    def execute(command, **kwargs):
        if command.startswith("test -f "):
            return "yes" if "/logs/verifier/reward.txt" in command else "no"
        if command == "cat /logs/verifier/reward.txt":
            return "1.0"
        if command == "cat /tmp/rllm-mini-swe-trajectory.json":
            return '{"info": {"exit_status": "Submitted"}}'
        return ""

    sandbox.exec.side_effect = execute
    _setup_task_environment(task, sandbox)
    evaluator = ShellScriptEvaluator(sandbox, verifier_timeout=600)
    out = evaluator.evaluate(task, Episode(id="episode", task="task", trajectories=[]))
    assert out.reward == 1.0
    assert MiniSweAgentHarness()._read_exit_outcome(sandbox)[0] == "Submitted"
    control_calls = [call for call in sandbox.exec.call_args_list if call.args[0].startswith(("mkdir ", "chmod 700 ", "chown ", "test -f ", "cat "))]
    assert len(control_calls) == 9
    assert all(call.kwargs["timeout"] == expected for call in control_calls)
    verifier_call = next(call for call in sandbox.exec.call_args_list if "/tests/test.sh" in call.args[0])
    assert verifier_call.kwargs["timeout"] == 600


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5", ""])
def test_invalid_control_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setenv("RLLM_SANDBOX_CONTROL_TIMEOUT_S", value)
    with pytest.raises(ValueError, match="must be a positive integer"):
        sandbox_control_timeout_s()
