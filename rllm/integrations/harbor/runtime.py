"""Unified Harbor runtime for both eval and training.

Single class that satisfies both the ``AgentFlow`` protocol (eval) and the
``RemoteAgentRuntime`` protocol (training).  All harbor task execution goes
through ``run_harbor_task()`` in trial_helper.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from rllm.integrations.harbor.trial_helper import (
    MODEL_PLACEHOLDER,
    HarborTaskOutcome,
    ensure_dummy_api_keys,
    outcome_to_episode,
    run_harbor_task,
    silence_harbor,
)

logger = logging.getLogger(__name__)


class HarborRuntime:
    """Unified Harbor runtime: runs harbor trials for both eval and training.

    For **eval** (``AgentFlow`` protocol):
        ``arun(task, config) -> Episode`` / ``run(task, config) -> Episode``

    For **training** (``RemoteAgentRuntime`` protocol):
        ``execute_tasks(submissions, timeout) -> list[RemoteTaskResult]``

    Both paths share the same ``run_harbor_task()`` core.
    """

    # Used by EvalRunner to cap concurrency.
    max_concurrent: int = 4

    def __init__(
        self,
        agent_name: str = "mini-swe-agent",
        environment_type: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        # Training-specific config (ignored by eval path).
        agent_timeout_multiplier: float | None = None,
        verifier_timeout_multiplier: float | None = None,
        agent_setup_timeout_multiplier: float | None = None,
        environment_build_timeout_multiplier: float | None = None,
        session_timeout: float = 900.0,
    ):
        self.agent_name = agent_name
        self.environment_type = environment_type
        self.agent_kwargs = agent_kwargs or {}
        self.agent_timeout_multiplier = agent_timeout_multiplier
        self.verifier_timeout_multiplier = verifier_timeout_multiplier
        self.agent_setup_timeout_multiplier = agent_setup_timeout_multiplier
        self.environment_build_timeout_multiplier = environment_build_timeout_multiplier
        self.session_timeout = session_timeout
        self._initialized = False

    # ------------------------------------------------------------------
    # Shared initialization
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        ensure_dummy_api_keys()
        silence_harbor()
        self._initialized = True

    def initialize(self) -> None:
        """Explicit initialization (``RemoteAgentRuntime`` protocol)."""
        import harbor.trial.trial  # noqa: F401

        self._ensure_initialized()
        logger.info("HarborRuntime initialized: agent=%s", self.agent_name)

    # ------------------------------------------------------------------
    # Core: run one task
    # ------------------------------------------------------------------

    async def _run_one(
        self,
        task_path: str,
        *,
        model_name: str | None = None,
        inference_url: str | None = None,
        trial_name: str = "",
        timeout: float | None = None,
    ) -> HarborTaskOutcome:
        """Run a single harbor task.  Shared by both eval and training."""
        self._ensure_initialized()
        return await run_harbor_task(
            task_path=task_path,
            agent_name=self.agent_name,
            model_name=model_name,
            inference_url=inference_url,
            environment_type=self.environment_type,
            agent_kwargs=self.agent_kwargs,
            agent_timeout_multiplier=self.agent_timeout_multiplier,
            verifier_timeout_multiplier=self.verifier_timeout_multiplier,
            agent_setup_timeout_multiplier=self.agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=self.environment_build_timeout_multiplier,
            trial_name=trial_name,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # AgentFlow protocol (eval)
    # ------------------------------------------------------------------

    async def arun(self, task, config) -> Episode:  # noqa: F821
        """Run a Harbor trial and return the result as an Episode.

        Satisfies the ``AgentFlow`` protocol for ``EvalRunner``.

        Args:
            task: rLLM Task wrapping a dict with ``task_path`` field.
            config: AgentConfig with ``base_url`` and ``model``.

        Returns:
            Episode with harbor_reward and harbor_is_correct in artifacts.
        """
        task_path = task.data.get("task_path")
        if not task_path:
            raise ValueError(f"Harbor task missing 'task_path' field in task data: {list(task.data.keys())}")

        outcome = await self._run_one(
            task_path=task_path,
            model_name=config.model,
            inference_url=config.base_url,
            trial_name=config.session_uid,
        )

        # Surface infrastructure failures as exceptions so EvalRunner counts
        # them as errors rather than silently reporting 0% accuracy.
        if not outcome.finished:
            raise RuntimeError(f"Harbor trial failed ({config.session_uid}): {outcome.error}")

        episode = outcome_to_episode(outcome, config.session_uid, task.data)

        # Store reward in artifacts so HarborEvaluator can read it.
        episode.artifacts["harbor_trial_ran"] = True
        episode.artifacts["harbor_reward"] = outcome.reward if outcome.reward is not None else 0.0
        episode.artifacts["harbor_is_correct"] = outcome.is_correct

        return episode

    def run(self, task, config) -> Episode:  # noqa: F821
        """Sync wrapper for arun."""
        return asyncio.run(self.arun(task, config))

    # ------------------------------------------------------------------
    # RemoteAgentRuntime protocol (training)
    # ------------------------------------------------------------------

    async def execute_tasks(
        self,
        submissions: list,
        timeout: float | None = None,
    ) -> list:
        """Submit tasks concurrently and gather results.

        Satisfies the ``RemoteAgentRuntime`` protocol.

        Args:
            submissions: list of ``TaskSubmission`` objects.
            timeout: Per-task timeout in seconds.

        Returns:
            list of ``RemoteTaskResult`` objects.
        """
        from rllm.experimental.engine.remote_runtime.protocol import RemoteTaskResult

        if not self._initialized:
            raise RuntimeError("Call initialize() before execute_tasks()")
        if timeout is None:
            timeout = self.session_timeout

        async def _run_submission(sub) -> RemoteTaskResult:
            task_path = sub.task.get("task_path")
            if not task_path:
                raise ValueError(f"Submission {sub.session_id} missing 'task_path' in task dict")

            outcome = await self._run_one(
                task_path=task_path,
                model_name=MODEL_PLACEHOLDER,
                inference_url=sub.inference_url,
                trial_name=sub.session_id,
                timeout=timeout,
            )

            meta = {"trial_uri": outcome.trial_uri} if outcome.trial_uri else {}
            return RemoteTaskResult(
                finished=outcome.finished,
                session_id=sub.session_id,
                task_id=sub.task_id,
                reward=outcome.reward,
                error=outcome.error,
                termination_reason=outcome.termination_reason,
                elapsed=outcome.elapsed,
                raw_result=outcome.raw_result,
                metadata=meta,
            )

        return list(await asyncio.gather(*[_run_submission(sub) for sub in submissions]))

    def shutdown(self) -> None:
        """Cleanup resources (``RemoteAgentRuntime`` protocol)."""
        self._initialized = False
        logger.info("HarborRuntime shutdown complete")
