"""Harbor-backed remote agent runtime.

Wraps harbor's ``Trial.create() / trial.run()`` API as a ``RemoteAgentRuntime``,
so rllm can drive SWE-smith (and any harbor-format task) through the same
interface used for the AgentCore backend.

Uses shared helpers from ``rllm.experimental.harbor.trial_helper`` for trial
config building, execution, and reward parsing — the same helpers used by
``HarborAgentFlow`` for eval.
"""

import asyncio
import logging
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

        from rllm.experimental.harbor.trial_helper import (
            ensure_dummy_api_keys,
            silence_harbor,
        )

        ensure_dummy_api_keys()
        silence_harbor()

        self._initialized = True
        logger.info("HarborRuntime initialized: agent=%s", self._h_config.agent)

    async def _run_one(self, sub: TaskSubmission, timeout: float) -> RemoteTaskResult:
        from rllm.experimental.harbor.trial_helper import (
            MODEL_PLACEHOLDER,
            build_harbor_trial_config,
            run_harbor_trial,
            trial_result_to_reward,
        )

        task_path = sub.task.get("task_path")
        if not task_path:
            raise ValueError(f"Submission {sub.session_id} missing 'task_path' in task dict")

        start = time.monotonic()
        try:
            trial_config = build_harbor_trial_config(
                task_path=task_path,
                agent_name=self._h_config.agent,
                model_name=MODEL_PLACEHOLDER,
                inference_url=sub.inference_url,
                environment_type=self._h_config.environment_type,
                agent_kwargs=dict(self._h_config.agent_kwargs),
                agent_timeout_multiplier=self._h_config.agent_timeout_multiplier,
                verifier_timeout_multiplier=self._h_config.verifier_timeout_multiplier,
                agent_setup_timeout_multiplier=self._h_config.agent_setup_timeout_multiplier,
                environment_build_timeout_multiplier=self._h_config.environment_build_timeout_multiplier,
                trial_name=sub.session_id,
            )
            result = await run_harbor_trial(trial_config, timeout=timeout)
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
        reward, is_correct, error_msg = trial_result_to_reward(result)
        exc_type = result.exception_info.exception_type if result.exception_info else None
        meta = {"trial_uri": result.trial_uri} if hasattr(result, "trial_uri") else {}

        if reward is not None:
            return RemoteTaskResult(
                finished=True,
                session_id=sub.session_id,
                task_id=sub.task_id,
                reward=reward,
                error=error_msg,
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
            error=error_msg or "harbor trial produced no verifier reward",
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
