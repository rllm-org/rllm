import asyncio
import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace

from rllm.integrations.harbor.eval_runner import run_harbor_eval
from rllm.types import Task


class _TrialResult:
    def __init__(self, path, reward):
        now = datetime.now()
        self.config = SimpleNamespace(task=SimpleNamespace(path=path))
        self.verifier_result = SimpleNamespace(rewards={"reward": reward})
        self.exception_info = None
        self.started_at = now
        self.finished_at = now + timedelta(seconds=2)
        self.trial_uri = None

    def model_dump(self, mode="json"):
        return {}


def test_harbor_job_owns_execution_and_converts_results(monkeypatch, tmp_path):
    from harbor.job import Job

    paths = [tmp_path / "a", tmp_path / "b"]
    for path in paths:
        path.mkdir()
    tasks = [Task(id=path.name, instruction="", metadata={"task_path": str(path)}, dataset_dir=path) for path in paths]
    # Deliberately completion-ordered rather than task-ordered.
    trials = [
        _TrialResult(paths[1], 0),
        _TrialResult(paths[0], 1),
        _TrialResult(paths[1], 1),
    ]
    harbor_config = tmp_path / "harbor.json"
    harbor_config.write_text(
        json.dumps(
            {
                "timeout_multiplier": 3.5,
                "environment": {"force_build": True},
                "agents": [{"name": "unused", "kwargs": {"reasoning_effort": "low"}}],
            }
        )
    )
    captured = {}

    class _Job:
        async def run(self):
            captured["runtime_env"] = {key: os.environ.get(key) for key in ("OPENROUTER_API_KEY", "LLM_API_KEY", "LLM_BASE_URL")}
            return SimpleNamespace(trial_results=trials)

    async def create(config):
        captured["config"] = config
        return _Job()

    monkeypatch.setattr(Job, "create", create)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result, episodes = asyncio.run(
        run_harbor_eval(
            tasks,
            agent_name="openhands-sdk",
            model="qwen/qwen3.8-27b",
            provider="openrouter",
            api_key="secret",
            base_url=None,
            concurrency=7,
            attempts=2,
            sandbox_backend="modal",
            dataset_name="swebench-verified",
            jobs_dir=str(tmp_path / "jobs"),
            agent_kwargs={"max_iterations": 42},
            agent_env={"EXPERIMENT": "direct"},
            harbor_config=str(harbor_config),
        )
    )

    config = captured["config"]
    assert (config.n_concurrent_trials, config.n_attempts) == (7, 2)
    assert config.timeout_multiplier == 3.5
    assert config.environment.type.value == "modal"
    assert config.environment.force_build is True
    assert config.jobs_dir == tmp_path / "jobs"
    assert config.agents[0].name == "openhands-sdk"
    assert config.agents[0].model_name == "openrouter/qwen/qwen3.8-27b"
    assert config.agents[0].kwargs == {"reasoning_effort": "low", "max_iterations": 42}
    assert config.agents[0].env == {"EXPERIMENT": "direct"}
    assert captured["runtime_env"] == {
        "OPENROUTER_API_KEY": "secret",
        "LLM_API_KEY": "secret",
        "LLM_BASE_URL": "https://openrouter.ai/api/v1",
    }
    assert os.environ.get("OPENROUTER_API_KEY") is None
    assert [episode.id for episode in episodes] == ["a:0", "b:0", "b:1"]
    assert [item.reward for item in result.items] == [1, 0, 0, 1]
    assert result.items[1].error == "missing Harbor trial"
    assert result.pass_at[2] == 1.0
