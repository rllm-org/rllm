"""CyberGymRuntime: Harbor trial + artifacts; infra failures raise."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rllm.integrations.harbor.cybergym.runtime import CyberGymRuntime
from rllm.integrations.harbor.trial_helper import HarborTaskOutcome
from rllm.types import Task, TerminationReason

from .cybergym_task_factory import write_cybergym_task


class _Cfg:
    model = "anthropic/claude-sonnet-4-5"
    base_url = "http://localhost:8000"
    session_uid = "sess-1"


def _outcome(*, finished=True, reward=1.0, error=None, stdout=""):
    return HarborTaskOutcome(
        finished=finished,
        reward=reward,
        is_correct=bool(reward and reward > 0),
        error=error,
        termination_reason=TerminationReason.ENV_DONE if finished else TerminationReason.ERROR,
        elapsed=1.0,
        raw_result={"verifier_result": {"stdout": stdout, "rewards": {"reward": reward}}},
        trial_uri=None,
    )


@pytest.mark.asyncio
async def test_arun_attaches_exit_codes(tmp_path):
    task_dir = write_cybergym_task(tmp_path)
    task = Task(id="t", instruction="x", metadata={"task_path": str(task_dir)}, dataset_dir=task_dir)
    runtime = CyberGymRuntime(agent_name="claude-code")
    stdout = """\
Verifying 1 PoC submission(s)
--- poc_001 ---
  vul_exit=139
  fix_exit=0
{"reward": 1.0, "vul_exit_code": 139, "fix_exit_code": 0}
"""

    async def _fake_run_one(**_kwargs):
        return _outcome(stdout=stdout)

    runtime._run_one = _fake_run_one  # type: ignore[method-assign]
    episode = await runtime.arun(task, _Cfg())
    assert episode.artifacts["cybergym_trial_ran"] is True
    assert episode.artifacts["cybergym_reward"] == 1.0
    assert episode.artifacts["vul_exit_code"] == 139
    assert episode.artifacts["fix_exit_code"] == 0
    assert episode.artifacts["n_submissions"] == 1
    assert episode.is_correct is True


@pytest.mark.asyncio
async def test_unfinished_trial_raises():
    task = Task(id="t", instruction="x", metadata={"task_path": "/tmp/missing-cybergym"}, dataset_dir=".")
    runtime = CyberGymRuntime(agent_name="claude-code")

    async def _fail(**_kwargs):
        return _outcome(finished=False, reward=None, error="sidecar down")

    runtime._run_one = _fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="sidecar down"):
        await runtime.arun(task, _Cfg())


@pytest.mark.asyncio
async def test_execute_tasks_logs_exits():
    runtime = CyberGymRuntime(agent_name="claude-code")
    runtime._initialized = True
    stdout = """{"reward": 0.0, "vul_exit_code": 0, "fix_exit_code": 0}"""

    async def _ok(**_kwargs):
        return _outcome(reward=0.0, stdout=stdout)

    runtime._run_one = _ok  # type: ignore[method-assign]
    sub = SimpleNamespace(
        session_id="s",
        task_id="t",
        inference_url="http://localhost:1",
        task={"task_path": "/tmp/x"},
    )
    results = await runtime.execute_tasks([sub], timeout=10)
    assert results[0].finished is True
    assert results[0].reward == 0.0
    assert results[0].metadata["vul_exit_code"] == 0
    assert results[0].metadata["n_submissions"] == 1
