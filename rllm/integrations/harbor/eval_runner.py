"""Run Harbor evals directly and adapt their artifacts to rLLM output."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from rllm.eval.results import EvalItem, EvalResult
from rllm.integrations.harbor.trial_helper import (
    HarborTaskOutcome,
    ensure_dummy_api_keys,
    map_termination_reason,
    outcome_to_episode,
    silence_harbor,
    trial_result_to_reward,
)


def _load_job_config(source: str | None, jobs_dir: str):
    from harbor.models.job.config import JobConfig

    if not source:
        return JobConfig(jobs_dir=Path(jobs_dir), quiet=True)

    path = Path(source).expanduser()
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Harbor config must contain a JSON/YAML object")
    data.setdefault("jobs_dir", jobs_dir)
    data.setdefault("quiet", True)
    return JobConfig.model_validate(data)


def _model_name(provider: str | None, model: str) -> str:
    if not provider or provider == "custom":
        return model
    from rllm.eval.config import get_provider_info

    info = get_provider_info(provider)
    prefix = info.litellm_prefix if info else provider
    return model if model.startswith(f"{prefix}/") else f"{prefix}/{model}"


def _runtime_env(provider: str | None, api_key: str | None, base_url: str | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if api_key:
        env["LLM_API_KEY"] = api_key
        if provider:
            from rllm.eval.config import get_provider_info

            info = get_provider_info(provider)
            if info and info.env_key:
                env[info.env_key] = api_key
    base_url = base_url or {"openrouter": "https://openrouter.ai/api/v1"}.get(provider or "")
    if base_url:
        env["LLM_BASE_URL"] = base_url
    return env


def _to_outcome(result) -> HarborTaskOutcome:
    reward, is_correct, error = trial_result_to_reward(result)
    exception_type = result.exception_info.exception_type if result.exception_info else None
    elapsed = 0.0
    if result.started_at and result.finished_at:
        elapsed = max(0.0, (result.finished_at - result.started_at).total_seconds())
    return HarborTaskOutcome(
        finished=reward is not None,
        reward=reward,
        is_correct=is_correct,
        error=error,
        termination_reason=map_termination_reason(reward is not None, exception_type),
        elapsed=elapsed,
        raw_result=result.model_dump(mode="json"),
        trial_uri=getattr(result, "trial_uri", None),
        _trial_result=result,
    )


async def run_harbor_eval(
    tasks: list,
    *,
    agent_name: str,
    model: str,
    provider: str | None,
    api_key: str | None,
    base_url: str | None,
    concurrency: int,
    attempts: int,
    sandbox_backend: str | None,
    dataset_name: str,
    jobs_dir: str,
    agent_kwargs: dict[str, Any] | None = None,
    agent_env: dict[str, str] | None = None,
    harbor_config: str | None = None,
    on_episode_complete=None,
):
    """Let one Harbor Job execute and verify all selected eval trials."""
    from harbor.job import Job
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import AgentConfig, TaskConfig

    task_paths = [Path(task.metadata["task_path"]).resolve() for task in tasks]
    config = _load_job_config(harbor_config, jobs_dir)
    agent_data = config.agents[0].model_dump() if config.agents else {}
    agent_data.update(name=agent_name, import_path=None, model_name=_model_name(provider, model))
    agent_data["kwargs"] = {**agent_data.get("kwargs", {}), **(agent_kwargs or {})}
    agent_data["env"] = {**agent_data.get("env", {}), **(agent_env or {})}

    environment = config.environment
    if sandbox_backend:
        environment = environment.model_copy(update={"type": EnvironmentType(sandbox_backend)})
    config = config.model_copy(
        update={
            "agents": [AgentConfig.model_validate(agent_data)],
            "datasets": [],
            "tasks": [TaskConfig(path=path) for path in task_paths],
            "n_attempts": attempts,
            "n_concurrent_trials": concurrency,
            "environment": environment,
        }
    )

    ensure_dummy_api_keys()
    silence_harbor()
    runtime_env = {**_runtime_env(provider, api_key, base_url), **agent_data["env"]}
    previous_env = {key: os.environ.get(key) for key in runtime_env}
    os.environ.update(runtime_env)
    try:
        result = await (await Job.create(config)).run()
    finally:
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    by_task: dict[Path, list] = defaultdict(list)
    for trial in result.trial_results:
        if trial.config.task.path:
            by_task[Path(trial.config.task.path).resolve()].append(trial)

    episodes, items = [], []
    for task_idx, (task, task_path) in enumerate(zip(tasks, task_paths, strict=True)):
        for attempt in range(attempts):
            trial = by_task[task_path].pop(0) if by_task[task_path] else None
            if trial is None:
                items.append(EvalItem(idx=task_idx, attempt=attempt, reward=0.0, is_correct=False, error="missing Harbor trial"))
                continue

            outcome = _to_outcome(trial)
            episode = outcome_to_episode(outcome, f"{task.id}:{attempt}", task.metadata)
            episode.artifacts.update(harbor_trial_ran=True, harbor_reward=outcome.reward or 0.0, harbor_is_correct=outcome.is_correct)
            if on_episode_complete:
                on_episode_complete(len(items), episode)
            items.append(
                EvalItem(
                    idx=task_idx,
                    attempt=attempt,
                    reward=outcome.reward or 0.0,
                    is_correct=outcome.is_correct,
                    error=None if outcome.finished else outcome.error,
                    termination_reason=outcome.termination_reason.value,
                )
            )
            if outcome.finished:
                episodes.append(episode)

    return EvalResult.from_items(dataset_name, model, f"harbor:{agent_name}", items, attempts=attempts), episodes
