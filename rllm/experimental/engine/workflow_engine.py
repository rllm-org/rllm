from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from rllm.engine.base import FlowEngine
from rllm.experimental.rollout import RolloutEngine
from rllm.types import Episode
from rllm.workflows.store import Store
from rllm.workflows.workflow import TerminationReason, Workflow

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rllm.utils.episode_logger import EpisodeLogger

logger = logging.getLogger(__name__)


class WorkflowEngine(FlowEngine):
    def __init__(
        self,
        workflow_cls: type[Workflow],
        workflow_args: dict,
        rollout_engine: RolloutEngine,
        config: DictConfig | None = None,
        n_parallel_tasks: int = 128,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger: EpisodeLogger | None = None,
        post_execute_hook: Callable[[], Coroutine[Any, Any, None]] | None = None,
        store: Store | None = None,
        **kwargs,
    ):
        """
        Initialize the WorkflowEngine.

        Args:
            workflow_cls: The workflow class to instantiate for each task.
            workflow_args: Arguments to pass to workflow instances.
            rollout_engine: Engine for model inference and rollout.
            config: Optional configuration object for training.
            n_parallel_tasks: Number of parallel workflow instances to maintain.
            retry_limit: Maximum number of retry attempts for failed tasks.
            raise_on_error: Whether to raise exceptions on permanent failures.
            episode_logger: Optional logger for saving episode data to files.
            post_execute_hook: Optional async callback invoked after all tasks
                in a batch complete but before results are returned.  Useful for
                batch-level trace flushing in SDK async-tracer mode.
            store: Optional cross-episode store shared across all workflow instances.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(n_parallel_tasks=n_parallel_tasks, episode_logger=episode_logger)
        self.workflow_cls = workflow_cls
        self.workflow_args = workflow_args or {}
        self.store = store

        self.rollout_engine = rollout_engine
        self.config = config  # if training

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.raise_on_error = raise_on_error
        self.kwargs = kwargs

        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tasks)
        self.workflow_queue: asyncio.Queue[Workflow] | None = None

        # Post-execute hook (e.g. SDK trace flushing)
        self.post_execute_hook = post_execute_hook

    async def initialize_pool(self):
        """Initialize the workflow pool with parallel workflow instances.

        Creates and populates the workflow queue with workflow instances
        for parallel task processing. This method is idempotent and will
        not recreate the pool if it already exists.
        """
        assert self.executor is not None, "executor is not initialized"
        if self.workflow_queue is not None:
            return
        logger.info(f"[WorkflowEngine] Initializing pool with {self.n_parallel_tasks} workflows")
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for _ in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(
                rollout_engine=self.rollout_engine,
                executor=self.executor,
                store=self.store,
                **self.workflow_args,
            )
            assert workflow.is_multithread_safe(), "Workflows must contain only thread-save environments"
            self.workflow_queue.put_nowait(workflow)
        logger.info(f"[WorkflowEngine] Pool initialized. Queue size: {self.workflow_queue.qsize()}")

    async def _pre_execute(self, *, is_validation: bool = False, **kwargs) -> None:
        if self.workflow_queue is None:
            await self.initialize_pool()
        self.rollout_engine.is_validation = is_validation

    async def _post_execute(self, *, is_validation: bool = False, **kwargs) -> None:
        if self.post_execute_hook is not None:
            await self.post_execute_hook()

    async def process_task_with_retry(
        self,
        task: dict,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        *,
        is_validation: bool = False,
        **kwargs,
    ) -> tuple[str, int, int, Episode]:
        """Process a single task rollout with retry logic based on termination reasons.

        Args:
            task: Task dictionary containing the task specification.
            task_id: Unique identifier for the task.
            rollout_idx: Index of this rollout attempt for the task.
            result_idx: Index of the result in the results list. This is useful for tracking the order of streaming results back.
            is_validation: Forwarded for parity with the base API; the per-batch
                value is set on ``rollout_engine`` in :meth:`_pre_execute`.
            **kwargs: Additional arguments passed to the workflow.

        Returns:
            tuple[str, int, int, Episode]: Task ID, rollout index, result index, and completed episode.

        Raises:
            Exception: If task fails permanently after retry_limit attempts and raise_on_error is True.
        """
        del is_validation  # already applied to rollout_engine in _pre_execute

        assert self.workflow_queue is not None, "workflow_queue is not initialized"
        logger.debug(f"[WorkflowEngine] Waiting for workflow from queue. Available: {self.workflow_queue.qsize()}")
        workflow = await self.workflow_queue.get()
        try:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                logger.debug(f"[WorkflowEngine] [{uid}] Starting attempt {retry_attempt}/{self.retry_limit}")
                workflow.reset(task=task, uid=uid)
                episode = await workflow.run_with_termination_handling(task=task, uid=uid, **kwargs)

                # We will make sure that the episode has the correct `uid` and `task` fields.
                episode.id = uid
                episode.task = task

                reward_strs = self.format_reward_strs(episode)
                logger.debug(f"[{uid}] Rollout completed. Rewards: [{', '.join(reward_strs)}], Termination: {episode.termination_reason}")

                if episode.termination_reason != TerminationReason.ERROR:
                    return task_id, rollout_idx, result_idx, episode

                error_tb = episode.info.get("error", {}).get("traceback")
                if error_tb:
                    logger.error(f"[WorkflowEngine] [{uid}] Error on attempt {retry_attempt}/{self.retry_limit}:\n{error_tb}")

                if retry_attempt < self.retry_limit:
                    logger.warning(f"[WorkflowEngine] [{uid}] Rollout failed on attempt {retry_attempt}/{self.retry_limit}, retrying...")
                    continue

            if not self.raise_on_error:
                logger.error(f"[WorkflowEngine] [{uid}] Rollout failed permanently after {self.retry_limit} attempts.")
            else:
                raise Exception(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts.")

            return task_id, rollout_idx, result_idx, episode

        finally:
            await self.workflow_queue.put(workflow)
            logger.debug(f"[WorkflowEngine] Returned workflow to queue. Available: {self.workflow_queue.qsize()}")
