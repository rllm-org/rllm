"""Gateway-free eval execution through Harbor's native Job runtime."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from rllm.eval.runner import episodes_to_eval_result
from rllm.integrations.harbor.trial_helper import (
    ensure_dummy_api_keys,
    harbor_result_to_outcome,
    outcome_to_episode,
    silence_harbor,
)


def _qualified_model(provider: str | None, model: str) -> str:
    """Return Harbor/LiteLLM's ``provider/model`` spelling."""
    if not provider or provider == "custom":
        return model if "/" in model else f"openai/{model}"

    from rllm.eval.config import get_provider_info

    info = get_provider_info(provider)
    prefix = info.litellm_prefix if info else provider
    return model if model.startswith(f"{prefix}/") else f"{prefix}/{model}"


def _provider_base_url(provider: str | None) -> str | None:
    """Use Harbor's provider registry for a public direct API endpoint."""
    if not provider or provider == "custom":
        return None

    try:
        from harbor.agents.model_connection import PROVIDERS
        from rllm.eval.config import get_provider_info

        aliases = {"gemini": "google", "together_ai": "together"}
        info = get_provider_info(provider)
        harbor_provider = info.litellm_prefix if info else provider
        access = PROVIDERS.get(aliases.get(harbor_provider, harbor_provider))
        return access.base_url if access else None
    except (ImportError, AttributeError):
        return None


def _configured_api_env(provider: str | None, api_key: str | None) -> dict[str, str]:
    """Build host-only credential env vars (never serialized in JobConfig)."""
    if not api_key:
        return {}

    env = {"LLM_API_KEY": api_key}
    if provider and provider != "custom":
        from rllm.eval.config import get_provider_info

        info = get_provider_info(provider)
        if info and info.env_key:
            env[info.env_key] = api_key

        # Harbor occasionally uses a canonical alias different from rLLM's.
        try:
            from harbor.agents.model_connection import PROVIDERS

            aliases = {"gemini": "google", "together_ai": "together"}
            harbor_provider = info.litellm_prefix if info else provider
            access = PROVIDERS.get(aliases.get(harbor_provider, harbor_provider))
            if access and access.api_key_envs:
                env[access.api_key_envs[0]] = api_key
        except (ImportError, AttributeError):
            pass
    else:
        env["OPENAI_API_KEY"] = api_key
    return env


def _agent_endpoint_env(base_url: str | None) -> dict[str, str]:
    if not base_url:
        return {}
    return {
        "LLM_BASE_URL": base_url,
        "OPENAI_API_BASE": base_url,
        "OPENAI_BASE_URL": base_url,
        "ANTHROPIC_BASE_URL": base_url,
        "OPENROUTER_BASE_URL": base_url,
    }


def _load_job_config(source: str | None, *, job_name: str, jobs_dir: str):
    from harbor.models.job.config import JobConfig

    if source:
        from harbor.cli.config_sources import load_config_source

        data = load_config_source(source)
        if not isinstance(data, dict):
            raise ValueError("Harbor config must contain a JSON/YAML object")
        return JobConfig.model_validate(data)
    return JobConfig(job_name=job_name, jobs_dir=Path(jobs_dir), quiet=True)


def _result_task_path(result) -> str | None:
    path = getattr(getattr(result, "config", None), "task", None)
    path = getattr(path, "path", None)
    return str(Path(path).resolve()) if path else None


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
    agent_timeout: int | None = None,
    harbor_config: str | None = None,
    on_episode_complete=None,
):
    """Run one Harbor Job and adapt its artifacts into rLLM eval output.

    There is deliberately no rLLM gateway, proxy, tunnel, AgentFlowEngine, or
    sandbox hook in this path. Harbor owns provisioning, scheduling, agent
    execution, verification, retries, and its native artifacts.
    """
    from harbor.job import Job
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

    task_paths: list[str] = []
    for task in tasks:
        task_path = task.metadata.get("task_path")
        if not task_path:
            raise ValueError(f"Harbor task {getattr(task, 'id', '<unknown>')} is missing metadata.task_path")
        task_paths.append(str(Path(task_path).resolve()))

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", dataset_name).strip("-") or "dataset"
    job_config = _load_job_config(harbor_config, job_name=f"rllm-{slug}-{uuid4().hex[:8]}", jobs_dir=jobs_dir)

    # A config file is an escape hatch for the complete Harbor JobConfig and
    # AgentConfig schemas. rLLM overrides only the fields defining this eval.
    base_agent = job_config.agents[0].model_dump() if harbor_config and job_config.agents else {}
    base_agent.update(name=agent_name, import_path=None, model_name=_qualified_model(provider, model))
    if agent_timeout is not None:
        base_agent["override_timeout_sec"] = agent_timeout
    base_agent["kwargs"] = {**base_agent.get("kwargs", {}), **(agent_kwargs or {})}
    endpoint = base_url or _provider_base_url(provider)
    base_agent["env"] = {**base_agent.get("env", {}), **_agent_endpoint_env(endpoint), **(agent_env or {})}

    environment = job_config.environment
    if sandbox_backend:
        from harbor.models.environment_type import EnvironmentType

        environment = EnvironmentConfig(**{**environment.model_dump(), "type": EnvironmentType(sandbox_backend)})

    job_config = job_config.model_copy(
        update={
            "agents": [AgentConfig.model_validate(base_agent)],
            "datasets": [],
            "tasks": [TaskConfig(path=Path(path)) for path in task_paths],
            "n_attempts": attempts,
            "n_concurrent_trials": concurrency,
            "environment": environment,
        }
    )

    ensure_dummy_api_keys()
    silence_harbor()
    configured_env = _configured_api_env(provider, api_key)
    previous_env = {key: os.environ.get(key) for key in configured_env}
    os.environ.update(configured_env)
    try:
        job = await Job.create(job_config)
        job_result = await job.run()
    finally:
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    # Harbor returns trials in completion order. Fold them back to rLLM's
    # task-major, attempt-minor convention so pass@k and episode filenames stay
    # stable. A task name fallback helps custom Harbor task resolvers.
    by_path: dict[str, list] = defaultdict(list)
    by_name: dict[str, list] = defaultdict(list)
    for result in job_result.trial_results:
        result_path = _result_task_path(result)
        if result_path:
            by_path[result_path].append(result)
        else:
            by_name[result.task_name].append(result)

    episodes: list = []
    for task, task_path in zip(tasks, task_paths, strict=True):
        candidates = by_path.get(task_path)
        if not candidates:
            candidates = by_name.get(Path(task_path).name, [])
        for attempt in range(attempts):
            result = candidates.pop(0) if candidates else None
            if result is None:
                episodes.append(None)
                continue
            outcome = harbor_result_to_outcome(result)
            episode = outcome_to_episode(outcome, f"{task.id}:{attempt}", task.metadata)
            episodes.append(episode)
            if on_episode_complete is not None:
                on_episode_complete(len(episodes) - 1, episode)

    return episodes_to_eval_result(
        episodes,
        dataset_name=dataset_name,
        model=model,
        agent_name=f"harbor:{agent_name}",
        attempts=attempts,
    )
