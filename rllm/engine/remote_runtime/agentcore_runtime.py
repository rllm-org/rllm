"""AgentCore Runtime adapter using ART's HTTP invocation contract."""

import asyncio
import logging
import time
from typing import Any

from rllm.engine.remote_runtime.protocol import (
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
        self._rate_limiter: Any = None

    def initialize(self) -> None:
        """Share one request budget across concurrent executions."""
        from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter

        cfg = self._acr_config
        self._rate_limiter = LocalRateLimiter(tps_limit=cfg.tps_limit)
        logger.info(
            "AgentCoreRuntime initialized: arn=%s, exp=%s, model=%s, tps=%d",
            cfg.agent_runtime_arn,
            self._exp_id,
            self._model_id,
            cfg.tps_limit,
        )

    async def _run_one(self, sub: TaskSubmission, timeout: float, client: Any) -> RemoteTaskResult:
        from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

        started = time.monotonic()
        raw_result = None
        try:
            async with AgentCoreHttpSession.with_client(sub.session_id, client) as session:
                async with asyncio.timeout(timeout):
                    result = await session.run(
                        {
                            **sub.task,
                            "_config": {
                                "base_url": sub.inference_url,
                                "model_id": self._model_id,
                                "api_key": sub.session_id,
                            },
                        }
                    )
                raw_result = result.task_output
                if error := result.failure_reason():
                    raise RuntimeError(error)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - started
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
            logger.error("Task %s failed: %s", sub.session_id, e)
            return RemoteTaskResult(
                finished=False,
                session_id=sub.session_id,
                task_id=sub.task_id,
                error=str(e),
                termination_reason=TerminationReason.ERROR,
                elapsed=time.monotonic() - started,
                raw_result=raw_result,
            )

        return RemoteTaskResult(
            finished=True,
            session_id=sub.session_id,
            task_id=sub.task_id,
            reward=result.reward,
            termination_reason=TerminationReason.ENV_DONE,
            elapsed=time.monotonic() - started,
            raw_result=raw_result,
        )

    async def execute_tasks(self, submissions: list[TaskSubmission], timeout: float | None = None) -> list[RemoteTaskResult]:
        """Share an HTTP client for this batch and close it after all sessions exit."""
        from agentcore_rl_toolkit import AgentCoreHttpClient

        if self._rate_limiter is None:
            raise RuntimeError("Call initialize() before execute_tasks()")

        if timeout is None:
            timeout = self._config.session_timeout

        client = AgentCoreHttpClient(
            self._acr_config.agent_runtime_arn,
            request_rate_limiter=self._rate_limiter,
        )
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self._run_one(sub, timeout, client)) for sub in submissions]
            return [task.result() for task in tasks]
        finally:
            await client.aclose()

    def shutdown(self) -> None:
        self._rate_limiter = None
        logger.info("AgentCoreRuntime shutdown complete")
