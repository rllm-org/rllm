"""Harbor Job runtime → rLLM eval artifact adapter tests."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from rllm.integrations.harbor.eval_runner import run_harbor_eval
from rllm.types import Task


class _TrialResult:
    def __init__(self, path: Path, reward: float):
        now = datetime.now()
        self.config = SimpleNamespace(task=SimpleNamespace(path=path))
        self.task_name = path.name
        self.verifier_result = SimpleNamespace(rewards={"reward": reward})
        self.exception_info = None
        self.started_at = now
        self.finished_at = now + timedelta(seconds=2)
        self.trial_uri = None

    def model_dump(self, mode="json"):
        return {"task_name": self.task_name}


def test_job_owns_scheduling_and_results_are_reordered(monkeypatch, tmp_path):
    from harbor.job import Job

    task_paths = [tmp_path / "task-a", tmp_path / "task-b"]
    for path in task_paths:
        path.mkdir()
    tasks = [Task(id=path.name, instruction="fix it", metadata={"task_path": str(path)}, dataset_dir=path) for path in task_paths]

    # Completion order is deliberately not rLLM's task-major result order.
    trial_results = [
        _TrialResult(task_paths[1], 0.0),
        _TrialResult(task_paths[0], 1.0),
        _TrialResult(task_paths[1], 1.0),
        _TrialResult(task_paths[0], 0.0),
    ]
    captured = {}

    class _FakeJob:
        async def run(self):
            captured["key_during_run"] = os.environ.get("OPENROUTER_API_KEY")
            return SimpleNamespace(trial_results=trial_results)

    async def _create(config):
        captured["config"] = config
        return _FakeJob()

    monkeypatch.setattr(Job, "create", _create)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    completed = []

    result, episodes = asyncio.run(
        run_harbor_eval(
            tasks,
            agent_name="openhands-sdk",
            model="qwen/qwen3.8-27b",
            provider="openrouter",
            api_key="or-secret",
            base_url=None,
            concurrency=7,
            attempts=2,
            sandbox_backend="modal",
            dataset_name="swebench-verified",
            jobs_dir=str(tmp_path / "jobs"),
            agent_kwargs={"max_iterations": 42},
            agent_env={"EXPERIMENT": "direct"},
            on_episode_complete=lambda idx, episode: completed.append((idx, episode)),
        )
    )

    config = captured["config"]
    assert config.n_concurrent_trials == 7
    assert config.n_attempts == 2
    assert config.environment.type.value == "modal"
    assert [task.path.resolve() for task in config.tasks] == task_paths
    assert config.agents[0].name == "openhands-sdk"
    assert config.agents[0].model_name == "openrouter/qwen/qwen3.8-27b"
    assert config.agents[0].kwargs == {"max_iterations": 42}
    assert config.agents[0].env["LLM_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert config.agents[0].env["EXPERIMENT"] == "direct"
    assert "OPENROUTER_API_KEY" not in config.agents[0].env
    assert captured["key_during_run"] == "or-secret"
    assert os.environ.get("OPENROUTER_API_KEY") is None

    assert [episode.id for episode in episodes] == ["task-a:0", "task-a:1", "task-b:0", "task-b:1"]
    assert [item.reward for item in result.items] == [1.0, 0.0, 0.0, 1.0]
    assert result.score == 0.5
    assert result.pass_at[2] == 1.0
    assert [idx for idx, _ in completed] == [0, 1, 2, 3]


def test_harbor_config_preserves_unmirrored_job_and_agent_fields(monkeypatch, tmp_path):
    from harbor.job import Job

    task_path = tmp_path / "task"
    task_path.mkdir()
    config_path = tmp_path / "job.json"
    config_path.write_text(
        json.dumps(
            {
                "job_name": "custom",
                "timeout_multiplier": 3.5,
                "retry": {"max_retries": 2},
                "environment": {"force_build": True},
                "agents": [
                    {
                        "name": "placeholder",
                        "skills": ["org/skill@v1"],
                        "kwargs": {"reasoning_effort": "low"},
                    }
                ],
            }
        )
    )
    captured = {}

    class _FakeJob:
        async def run(self):
            return SimpleNamespace(trial_results=[_TrialResult(task_path, 1.0)])

    async def _create(config):
        captured["config"] = config
        return _FakeJob()

    monkeypatch.setattr(Job, "create", _create)
    task = Task(id="task", instruction="", metadata={"task_path": str(task_path)}, dataset_dir=task_path)

    asyncio.run(
        run_harbor_eval(
            [task],
            agent_name="openhands-sdk",
            model="openrouter/qwen/qwen3.8-27b",
            provider=None,
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
            concurrency=1,
            attempts=1,
            sandbox_backend=None,
            dataset_name="bench",
            jobs_dir=str(tmp_path / "jobs"),
            agent_kwargs={"max_iterations": 10},
            harbor_config=str(config_path),
        )
    )

    config = captured["config"]
    assert config.timeout_multiplier == 3.5
    assert config.retry.max_retries == 2
    assert config.environment.force_build is True
    assert config.agents[0].skills == ["org/skill@v1"]
    assert config.agents[0].kwargs == {"reasoning_effort": "low", "max_iterations": 10}
