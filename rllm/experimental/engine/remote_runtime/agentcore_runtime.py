"""AgentCore Runtime adapter wrapping ART's RolloutClient async mode.

Implements RemoteAgentRuntime for AWS Bedrock AgentCore Runtime. Uses
``RolloutClient.invoke_async()`` for concurrent task submission and
``RolloutFuture.result_async()`` for S3 reward polling.
"""

import asyncio
import logging
from typing import Any

from rllm.experimental.engine.remote_runtime.protocol import (
    AgentCoreRuntimeConfig,
    RemoteAgentRuntime,
    RemoteRuntimeConfig,
    RemoteTaskResult,
    TaskSubmission,
)
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)


class AgentCoreRuntime(RemoteAgentRuntime):
    """Remote agent runtime backed by AWS Bedrock AgentCore."""

    def __init__(
        self,
        config: RemoteRuntimeConfig,
        exp_id: str = "",
        model_id: str = "",
    ) -> None:
        self._config = config
        self._acr_config = AgentCoreRuntimeConfig(**config.agentcore)
        self._exp_id = exp_id
        self._model_id = model_id
        self._client: Any = None

    def initialize(self) -> None:
        """Create the ART RolloutClient from config."""
        from agentcore_rl_toolkit import RolloutClient

        cfg = self._acr_config
        self._client = RolloutClient(
            agent_runtime_arn=cfg.agent_runtime_arn,
            s3_bucket=cfg.s3_bucket,
            exp_id=self._exp_id,
            tps_limit=cfg.tps_limit,
        )
        logger.info(
            "AgentCoreRuntime initialized: arn=%s, bucket=%s, exp=%s, model=%s, tps=%d",
            cfg.agent_runtime_arn,
            cfg.s3_bucket,
            self._exp_id,
            self._model_id,
            cfg.tps_limit,
        )

    async def _run_one(self, sub: TaskSubmission, timeout: float) -> RemoteTaskResult:
        """Submit a single task and poll S3 for the result."""
        try:
            future = await self._client.invoke_async(
                payload=sub.task,
                session_id=sub.session_id,
                input_id=sub.task_id,
                base_url=sub.inference_url,
                model_id=self._model_id,
            )
        except Exception as e:
            logger.error("Task %s submission failed: %s", sub.session_id, e)
            return RemoteTaskResult(
                finished=False,
                session_id=sub.session_id,
                task_id=sub.task_id,
                error=str(e),
                termination_reason=TerminationReason.ERROR,
                elapsed=0.0,
            )

        try:
            result = await future.result_async(timeout=timeout)
        except asyncio.TimeoutError:
            elapsed = future.elapsed()
            logger.error("Task %s timed out after %.1fs", sub.session_id, elapsed)
            return RemoteTaskResult(
                finished=False,
                session_id=sub.session_id,
                task_id=sub.task_id,
                error=f"Timeout after {elapsed:.1f}s",
                termination_reason=TerminationReason.TIMEOUT,
                elapsed=elapsed,
            )
        except Exception as e:
            elapsed = future.elapsed()
            logger.error("Task %s polling failed: %s", sub.session_id, e)
            return RemoteTaskResult(
                finished=False,
                session_id=sub.session_id,
                task_id=sub.task_id,
                error=str(e),
                termination_reason=TerminationReason.ERROR,
                elapsed=elapsed,
            )

        # Check for application-level errors from @rollout_entrypoint in ART
        status_code = result.get("status_code")
        if status_code is not None and status_code != 200:
            error_msg = result.get("stop_reason", "unknown remote error")
            logger.error(
                "Task %s returned status_code=%s: %s",
                sub.session_id,
                status_code,
                error_msg,
            )
            return RemoteTaskResult(
                finished=False,
                session_id=sub.session_id,
                task_id=sub.task_id,
                error=error_msg,
                termination_reason=TerminationReason.ERROR,
                elapsed=future.elapsed(),
                raw_result=result,
            )

        reward = result.get("rewards")
        if isinstance(reward, list):
            reward = reward[-1] if reward else None

        return RemoteTaskResult(
            finished=True,
            session_id=sub.session_id,
            task_id=sub.task_id,
            reward=reward,
            termination_reason=TerminationReason.ENV_DONE,
            elapsed=future.elapsed(),
            raw_result=result,
        )

    async def execute_tasks(self, submissions: list[TaskSubmission], timeout: float | None = None) -> list[RemoteTaskResult]:
        """Submit all tasks concurrently via asyncio.gather.

        Each task invokes then polls in sequence; all tasks run in parallel.
        ART rate-limits invocations at TPS internally.
        """
        if self._client is None:
            raise RuntimeError("Call initialize() before execute_tasks()")

        if timeout is None:
            timeout = self._config.session_timeout

        return list(await asyncio.gather(*[self._run_one(sub, timeout) for sub in submissions]))

    def shutdown(self) -> None:
        logger.info("AgentCoreRuntime shutdown complete")
