"""Harbor-backed remote agent runtime.

Wraps harbor's ``Trial.create() / trial.run()`` API as a ``RemoteAgentRuntime``,
so rllm can drive SWE-smith (and any harbor-format task) through the same
interface used for the AgentCore backend.

v1 is local-docker-only and eval-focused: no gateway, no training loop.
"""

import asyncio
import logging
import os
import time

from rllm.experimental.engine.remote_runtime.protocol import (
    HarborRuntimeConfig,
    RemoteAgentRuntime,
    RemoteRuntimeConfig,
    RemoteTaskResult,
    TaskSubmission,
)
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)

# Map harbor exception_type strings to rllm TerminationReason.
_EXCEPTION_TYPE_MAP: dict[str, TerminationReason] = {
    "AgentTimeoutError": TerminationReason.TIMEOUT,
    "ContextLengthExceededError": TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED,
    "OutputLengthExceededError": TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED,
}


def _map_termination_reason(
    finished: bool,
    exception_type: str | None,
    timed_out: bool = False,
) -> TerminationReason:
    """Derive TerminationReason from harbor trial outcome."""
    if finished:
        return TerminationReason.ENV_DONE
    if timed_out:
        return TerminationReason.TIMEOUT
    if exception_type:
        return _EXCEPTION_TYPE_MAP.get(exception_type, TerminationReason.ERROR)
    return TerminationReason.ERROR


# Provider API key env vars that harbor's installed agent scaffolds pre-flight
# check on the host process before the trial runs. When routing through a
# gateway the actual key value is irrelevant — vLLM doesn't validate it — but
# the var must exist or harbor raises. We set dummies for any that are unset,
# leaving real values alone if the user did set them.
_DUMMY_API_KEYS = {
    "OPENAI_API_KEY": "empty",  # mini-swe-agent, codex, swe-agent, qwen-coder
    "ANTHROPIC_API_KEY": "empty",  # claude-code
    "LLM_API_KEY": "empty",  # openhands, openhands-sdk (litellm)
}

# Placeholder model fed to harbor, which get's overridden by gateway
_HARBOR_MODEL_PLACEHOLDER = "openai/placeholder"


_HARBOR_FILTER_ATTR = "_rllm_drop_harbor_installed"


def silence_harbor() -> None:
    """Drop every log record from any harbor-namespaced logger.

    Harbor pins child loggers to DEBUG via its own ``setup_logger`` and uses
    dynamic logger names that include trial UUIDs, so a level filter on the
    parent doesn't help. We install a filter on every root handler that drops
    records whose logger name contains ``harbor``. Idempotent — safe to call
    repeatedly. Reusable from any entry point that imports harbor.
    """

    class _DropHarbor(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "harbor" not in record.name

    for h in logging.root.handlers:
        if getattr(h, _HARBOR_FILTER_ATTR, False):
            continue
        h.addFilter(_DropHarbor())
        setattr(h, _HARBOR_FILTER_ATTR, True)


class HarborRuntime(RemoteAgentRuntime):
    """Remote agent runtime backed by the harbor framework."""

    def __init__(
        self,
        config: RemoteRuntimeConfig,
    ) -> None:
        self._config = config
        self._h_config = HarborRuntimeConfig(**config.harbor)
        self._initialized = False

    def initialize(self) -> None:
        # Lazy import — keeps harbor off the rllm import path for other backends.
        import harbor.trial.trial  # noqa: F401

        # Satisfy harbor scaffolds' provider-key pre-flight checks. Real keys
        # (if the user set them) are left alone; only missing ones get dummies.
        for key, dummy in _DUMMY_API_KEYS.items():
            if key not in os.environ:
                os.environ[key] = dummy

        # Drop harbor's noisy DEBUG output from the host logging stream.
        silence_harbor()

        self._initialized = True
        logger.info("HarborRuntime initialized: agent=%s", self._h_config.agent)

    def _build_trial_config(self, sub: TaskSubmission):
        from pathlib import Path

        from harbor.models.environment_type import EnvironmentType
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            TrialConfig,
        )

        task_path = sub.task.get("task_path")
        if not task_path:
            raise ValueError(f"Submission {sub.session_id} missing 'task_path' in task dict")

        env: dict[str, str] = {}
        if sub.inference_url:
            # Set the inference base URL under every name harbor's installed
            # agent scaffolds read. Each scaffold checks a different one:
            #   OPENAI_API_BASE   — mini-swe-agent (also openhands fallback)
            #   OPENAI_BASE_URL   — codex, hermes, qwen-coder, swe-agent
            #   LLM_BASE_URL      — openhands, openhands-sdk
            #   ANTHROPIC_BASE_URL — claude-code
            env["OPENAI_API_BASE"] = sub.inference_url
            env["OPENAI_BASE_URL"] = sub.inference_url
            env["LLM_BASE_URL"] = sub.inference_url
            env["ANTHROPIC_BASE_URL"] = sub.inference_url

        env_type = EnvironmentType(self._h_config.environment_type) if self._h_config.environment_type else None

        return TrialConfig(
            task=TaskConfig(path=Path(task_path)),
            trial_name=sub.session_id,
            agent_timeout_multiplier=self._h_config.agent_timeout_multiplier,
            verifier_timeout_multiplier=self._h_config.verifier_timeout_multiplier,
            agent_setup_timeout_multiplier=self._h_config.agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=self._h_config.environment_build_timeout_multiplier,
            agent=AgentConfig(
                name=self._h_config.agent,
                model_name=_HARBOR_MODEL_PLACEHOLDER,
                kwargs=dict(self._h_config.agent_kwargs),
                env=env,
            ),
            environment=EnvironmentConfig(type=env_type),
        )

    async def _run_one(self, sub: TaskSubmission, timeout: float) -> RemoteTaskResult:
        from harbor.trial.trial import Trial

        start = time.monotonic()
        try:
            trial_config = self._build_trial_config(sub)
            trial = await Trial.create(trial_config)
            result = await asyncio.wait_for(trial.run(), timeout=timeout)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            logger.warning("Task %s timed out after %.1fs", sub.session_id, elapsed)
            return RemoteTaskResult(
                finished=False,
                session_id=sub.session_id,
                task_id=sub.task_id,
                error=f"harbor trial timed out after {timeout:.1f}s",
                termination_reason=TerminationReason.TIMEOUT,
                elapsed=elapsed,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.exception("Task %s failed: %s", sub.session_id, e)
            return RemoteTaskResult(
                finished=False,
                session_id=sub.session_id,
                task_id=sub.task_id,
                error=f"{type(e).__name__}: {e}",
                termination_reason=_map_termination_reason(False, type(e).__name__),
                elapsed=elapsed,
            )

        elapsed = time.monotonic() - start
        raw = result.model_dump(mode="json")
        vr = result.verifier_result
        exc = result.exception_info
        exc_msg = f"{exc.exception_type}: {exc.exception_message}" if exc else None
        exc_type = exc.exception_type if exc else None
        meta = {"trial_uri": result.trial_uri}

        if vr is not None and vr.rewards:
            reward_val = vr.rewards.get("reward")
            if reward_val is None:
                # Verifier emitted a non-"reward" key — fall back to the first.
                reward_val = next(iter(vr.rewards.values()))
            if reward_val is not None:
                return RemoteTaskResult(
                    finished=True,
                    session_id=sub.session_id,
                    task_id=sub.task_id,
                    reward=float(reward_val),
                    error=exc_msg,
                    termination_reason=_map_termination_reason(True, exc_type),
                    elapsed=elapsed,
                    raw_result=raw,
                    metadata=meta,
                )

        # No reward signal at all → finished=False.
        return RemoteTaskResult(
            finished=False,
            session_id=sub.session_id,
            task_id=sub.task_id,
            reward=None,
            error=exc_msg or "harbor trial produced no verifier reward",
            termination_reason=_map_termination_reason(False, exc_type),
            elapsed=elapsed,
            raw_result=raw,
            metadata=meta,
        )

    async def execute_tasks(
        self,
        submissions: list[TaskSubmission],
        timeout: float | None = None,
    ) -> list[RemoteTaskResult]:
        if not self._initialized:
            raise RuntimeError("Call initialize() before execute_tasks()")
        if timeout is None:
            timeout = self._config.session_timeout
        return list(await asyncio.gather(*[self._run_one(sub, timeout) for sub in submissions]))

    def shutdown(self) -> None:
        self._initialized = False
        logger.info("HarborRuntime shutdown complete")
