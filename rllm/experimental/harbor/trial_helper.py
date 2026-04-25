"""Shared Harbor trial execution helpers.

Contains the core logic for building TrialConfigs, running trials, and parsing
results. Used by both HarborAgentFlow (eval) and HarborRuntime (training) to
avoid duplicating Harbor interaction code.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Provider API key env vars that harbor's installed agent scaffolds pre-flight
# check on the host process before the trial runs. When routing through a
# proxy/gateway the actual key value is irrelevant, but the var must exist or
# harbor raises. We set dummies for any that are unset, leaving real values
# alone if the user did set them.
_DUMMY_API_KEYS = {
    "OPENAI_API_KEY": "empty",
    "ANTHROPIC_API_KEY": "empty",
    "LLM_API_KEY": "empty",
}

# Placeholder model when the real model is irrelevant (training via gateway)
MODEL_PLACEHOLDER = "openai/placeholder"

_HARBOR_FILTER_ATTR = "_rllm_drop_harbor_installed"


def ensure_dummy_api_keys() -> None:
    """Set dummy API keys for Harbor agent pre-flight checks.

    Harbor scaffolds check ``os.environ`` for provider keys before running.
    When routing through a proxy/gateway the actual value doesn't matter,
    but the variable must exist. Real user-set values are left alone.
    """
    for key, dummy in _DUMMY_API_KEYS.items():
        if key not in os.environ:
            os.environ[key] = dummy


def silence_harbor() -> None:
    """Drop every log record from any harbor-namespaced logger.

    Harbor pins child loggers to DEBUG via its own ``setup_logger`` and uses
    dynamic logger names that include trial UUIDs, so a level filter on the
    parent doesn't help. We install a filter on every root handler that drops
    records whose logger name contains ``harbor``. Idempotent.
    """

    class _DropHarbor(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "harbor" not in record.name

    for h in logging.root.handlers:
        if getattr(h, _HARBOR_FILTER_ATTR, False):
            continue
        h.addFilter(_DropHarbor())
        setattr(h, _HARBOR_FILTER_ATTR, True)


def _rewrite_url_for_container(url: str) -> str:
    """Rewrite localhost URLs so they're reachable from inside a Docker container.

    On macOS/Windows Docker Desktop, ``host.docker.internal`` resolves to the
    host machine. On Linux with default bridge networking, the same hostname
    is available in Docker 20.10+.
    """
    import re

    # Match http://127.0.0.1:PORT or http://localhost:PORT (with optional path)
    return re.sub(
        r"(https?://)(?:127\.0\.0\.1|localhost)(:\d+)",
        r"\1host.docker.internal\2",
        url,
    )


def _infer_provider_prefix(model_name: str) -> str:
    """Add a provider/ prefix to a bare model name.

    Harbor agents require ``provider/model`` format. This infers the provider
    from common model name patterns. Falls back to ``openai/`` if unknown.
    """
    name_lower = model_name.lower()
    if any(k in name_lower for k in ("claude", "haiku", "sonnet", "opus")):
        return f"anthropic/{model_name}"
    if any(k in name_lower for k in ("gemini", "gemma")):
        return f"google/{model_name}"
    if any(k in name_lower for k in ("qwen", "deepseek")):
        return f"openai/{model_name}"
    # Default to openai — works for gpt-*, o1-*, etc. and also when going
    # through a LiteLLM proxy which accepts openai/ for any backend.
    return f"openai/{model_name}"


def build_harbor_trial_config(
    task_path: str,
    agent_name: str,
    model_name: str | None = None,
    inference_url: str | None = None,
    environment_type: str | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    agent_timeout_multiplier: float | None = None,
    verifier_timeout_multiplier: float | None = None,
    agent_setup_timeout_multiplier: float | None = None,
    environment_build_timeout_multiplier: float | None = None,
    trial_name: str = "",
):
    """Build a Harbor TrialConfig from rLLM task data.

    This is the single source of truth for constructing trial configs, used
    by both eval (HarborAgentFlow) and training (HarborRuntime).

    Args:
        task_path: Absolute path to a Harbor task directory.
        agent_name: Harbor agent name (e.g., "mini-swe-agent", "claude-code").
        model_name: LLM model identifier (e.g., "anthropic/claude-opus-4-1").
            For training via gateway, use MODEL_PLACEHOLDER.
        inference_url: Base URL for LLM API calls. For eval this is rLLM's
            proxy or direct endpoint. For training this is the gateway URL.
        environment_type: Harbor environment backend (e.g., "docker", "daytona").
            None uses the task's default.
        agent_kwargs: Extra kwargs passed through to the Harbor agent scaffold.
        agent_timeout_multiplier: Multiply agent timeout by this factor.
        verifier_timeout_multiplier: Multiply verifier timeout by this factor.
        agent_setup_timeout_multiplier: Multiply agent setup timeout.
        environment_build_timeout_multiplier: Multiply environment build timeout.
        trial_name: Unique trial identifier. Auto-generated if empty.

    Returns:
        A ``harbor.models.trial.config.TrialConfig`` instance.
    """
    from pathlib import Path

    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )

    # Ensure model_name has provider/ prefix (Harbor agents require it)
    if model_name and "/" not in model_name:
        model_name = _infer_provider_prefix(model_name)

    env: dict[str, str] = {}
    if inference_url:
        # Rewrite localhost URLs to host.docker.internal so the agent
        # inside a Docker container can reach the host's proxy/gateway.
        container_url = _rewrite_url_for_container(inference_url)

        # Set the inference base URL under every name harbor's installed
        # agent scaffolds read. Each scaffold checks a different one:
        #   OPENAI_API_BASE   — mini-swe-agent (also openhands fallback)
        #   OPENAI_BASE_URL   — codex, hermes, qwen-coder, swe-agent
        #   LLM_BASE_URL      — openhands, openhands-sdk
        #   ANTHROPIC_BASE_URL — claude-code
        env["OPENAI_API_BASE"] = container_url
        env["OPENAI_BASE_URL"] = container_url
        env["LLM_BASE_URL"] = container_url
        env["ANTHROPIC_BASE_URL"] = container_url

    env_type = None
    if environment_type:
        from harbor.models.environment_type import EnvironmentType

        env_type = EnvironmentType(environment_type)

    return TrialConfig(
        task=TaskConfig(path=Path(task_path)),
        trial_name=trial_name,
        agent_timeout_multiplier=agent_timeout_multiplier,
        verifier_timeout_multiplier=verifier_timeout_multiplier,
        agent_setup_timeout_multiplier=agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=environment_build_timeout_multiplier,
        agent=AgentConfig(
            name=agent_name,
            model_name=model_name or MODEL_PLACEHOLDER,
            kwargs=dict(agent_kwargs) if agent_kwargs else {},
            env=env,
        ),
        environment=EnvironmentConfig(type=env_type),
    )


async def run_harbor_trial(trial_config, timeout: float | None = None):
    """Run a Harbor trial and return the TrialResult.

    Args:
        trial_config: A ``harbor.models.trial.config.TrialConfig``.
        timeout: Maximum time in seconds. None means no timeout.

    Returns:
        A ``harbor.models.trial.result.TrialResult``.
    """
    from harbor.trial.trial import Trial

    trial = await Trial.create(trial_config)
    if timeout is not None:
        return await asyncio.wait_for(trial.run(), timeout=timeout)
    return await trial.run()


def trial_result_to_reward(result) -> tuple[float | None, bool, str | None]:
    """Extract (reward, is_correct, error_message) from a Harbor TrialResult.

    Args:
        result: A ``harbor.models.trial.result.TrialResult``.

    Returns:
        Tuple of (reward float or None, is_correct bool, error string or None).
    """
    exc = result.exception_info
    error_msg = f"{exc.exception_type}: {exc.exception_message}" if exc else None

    vr = result.verifier_result
    if vr is not None and vr.rewards:
        reward_val = vr.rewards.get("reward")
        if reward_val is None:
            # Verifier emitted a non-"reward" key — fall back to the first value.
            reward_val = next(iter(vr.rewards.values()))
        if reward_val is not None:
            reward = float(reward_val)
            return reward, reward > 0, error_msg

    return None, False, error_msg or "harbor trial produced no verifier reward"


def trial_result_to_episode(result, uid: str, task: dict):
    """Convert a Harbor TrialResult into an rLLM Episode.

    Args:
        result: A ``harbor.models.trial.result.TrialResult``.
        uid: Unique episode identifier.
        task: The original task dict (from the dataset row).

    Returns:
        An ``rllm.types.Episode`` populated with reward and metadata.
    """
    from rllm.agents.agent import Episode, Trajectory
    from rllm.experimental.harbor.atif_trajectory_bridge import load_atif_steps
    from rllm.workflows.workflow import TerminationReason

    reward, is_correct, error_msg = trial_result_to_reward(result)

    # Map Harbor exception types to rLLM termination reasons
    _EXCEPTION_TYPE_MAP = {
        "AgentTimeoutError": TerminationReason.TIMEOUT,
        "ContextLengthExceededError": TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED,
        "OutputLengthExceededError": TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED,
    }

    exc_type = result.exception_info.exception_type if result.exception_info else None
    if reward is not None:
        termination = TerminationReason.ENV_DONE
    elif exc_type:
        termination = _EXCEPTION_TYPE_MAP.get(exc_type, TerminationReason.ERROR)
    else:
        termination = TerminationReason.ERROR

    # Load ATIF trajectory steps from disk if available.
    steps = []
    trial_uri = getattr(result, "trial_uri", None)
    if trial_uri:
        steps = load_atif_steps(trial_uri)

    trajectories = []
    if reward is not None:
        trajectories.append(
            Trajectory(
                name="harbor_trial",
                task=task,
                steps=steps,
                reward=reward,
            )
        )

    metrics = {
        "reward": reward if reward is not None else 0.0,
        "is_correct": int(is_correct),
    }

    # Extract timing info if available
    if hasattr(result, "started_at") and result.started_at and hasattr(result, "finished_at") and result.finished_at:
        elapsed = (result.finished_at - result.started_at).total_seconds()
        metrics["elapsed_sec"] = elapsed

    metadata: dict[str, Any] = {}
    if hasattr(result, "trial_uri") and result.trial_uri:
        metadata["trial_uri"] = result.trial_uri
    if error_msg:
        metadata["error"] = {"message": error_msg}

    return Episode(
        id=uid,
        task=task,
        is_correct=is_correct,
        trajectories=trajectories,
        metrics=metrics,
        metadata=metadata,
        termination_reason=termination,
    )
