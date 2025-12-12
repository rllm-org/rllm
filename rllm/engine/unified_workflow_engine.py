from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from tqdm import tqdm

from rllm.agents.agent import Episode
from rllm.engine.rollout import RolloutEngine, VerlEngine
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason, Workflow

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from rllm.utils.episode_logger import EpisodeLogger
    from verl import DataProto

logger = logging.getLogger(__name__)


class UnifiedWorkflowEngine:
    def __init__(
        self,
        workflow_cls: type[Workflow],
        workflow_args: dict,
        rollout_engine: RolloutEngine,
        config=None,
        n_parallel_tasks: int = 128,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger: EpisodeLogger | None = None,
        **kwargs,
    ):
        """
        Initialize the UnifiedWorkflowEngine.

        Args:
            workflow_cls: The workflow class to instantiate for each task.
            workflow_args: Arguments to pass to workflow instances.
            rollout_engine: Engine for model inference and rollout.
            config: Optional configuration object for training.
            n_parallel_tasks: Number of parallel workflow instances to maintain.
            retry_limit: Maximum number of retry attempts for failed tasks.
            raise_on_error: Whether to raise exceptions on permanent failures.
            episode_logger: Optional logger for saving episode data to files.
            **kwargs: Additional keyword arguments.
        """
        self.workflow_cls = workflow_cls
        self.workflow_args = workflow_args or {}

        self.rollout_engine = rollout_engine
        self.config = config  # if training

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.raise_on_error = raise_on_error
        self.kwargs = kwargs

        self.n_parallel_tasks = n_parallel_tasks
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tasks)
        self.workflow_queue = None

        # Episode logging support
        self.episode_logger = episode_logger
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"  # "train" or "val"

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
        """Set current training step for episode logging.

        Args:
            step: Current training step number
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    async def initialize_pool(self):
        """Initialize the workflow pool with parallel workflow instances.

        Creates and populates the workflow queue with workflow instances
        for parallel task processing. This method is idempotent and will
        not recreate the pool if it already exists.
        """
        assert self.executor is not None, "executor is not initialized"
        if self.workflow_queue is not None:
            return
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for i in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(rollout_engine=self.rollout_engine, executor=self.executor, **self.workflow_args)
            assert workflow.is_multithread_safe(), "Workflows must contain only thread-save environments"
            self.workflow_queue.put_nowait(workflow)

    async def process_task_with_retry(self, task: dict, task_id: str, rollout_idx: int, result_idx: int, **kwargs) -> tuple[str, int, int, Episode]:
        """Process a single task rollout with retry logic based on termination reasons.

        Args:
            task: Task dictionary containing the task specification.
            task_id: Unique identifier for the task.
            rollout_idx: Index of this rollout attempt for the task.
            result_idx: Index of the result in the results list. This is useful for tracking the order of streaming results back.
            **kwargs: Additional arguments passed to the workflow.

        Returns:
            tuple[str, int, int, Episode]: Task ID, rollout index, result index, and completed episode.

        Raises:
            Exception: If task fails permanently after retry_limit attempts and raise_on_error is True.
        """
        assert self.workflow_queue is not None, "workflow_queue is not initialized"
        workflow = await self.workflow_queue.get()
        try:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                episode = await workflow.run_with_termination_handling(task=task, uid=uid, **kwargs)

                # We will make sure that the episode has the correct `uid` and `task` fields.
                episode.id = uid
                episode.task = task

                # Display rewards for all trajectories. Fallback to last step reward if trajectory reward is not set.
                reward_strs = []
                for traj in episode.trajectories:
                    reward = "N/A"
                    if traj.reward is not None:
                        reward = f"{traj.reward:.1f}"
                    elif len(traj.steps) > 0:
                        reward = f"{traj.steps[-1].reward:.1f}"
                    reward_strs.append(f"{traj.name}: {reward}")
                colorful_print(f"[{uid}] Rollout completed. Rewards: [{', '.join(reward_strs)}], Termination: {episode.termination_reason}", fg="green" if episode.is_correct else "yellow")

                if episode.termination_reason != TerminationReason.ERROR:
                    return task_id, rollout_idx, result_idx, episode

                error_tb = episode.info.get("error", {}).get("traceback")
                if error_tb:
                    print(error_tb)

                if retry_attempt < self.retry_limit:
                    print(f"[{uid}] Rollout failed on attempt {retry_attempt}/{self.retry_limit}, retrying...")
                    continue

            if not self.raise_on_error:
                print(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts.")
            else:
                raise Exception(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts.")

            return task_id, rollout_idx, episode

        finally:
            await self.workflow_queue.put(workflow)

    async def execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
        """Run asynchronous workflow execution with retry logic for multiple tasks.
        TODO(listar2000): refactor this function to get rid of the ugly `task_states` dictionary.
        Args:
            tasks: List of task dictionaries to process.
            task_ids: Optional list of task identifiers. If None, UUIDs are generated.
            **kwargs: Additional arguments passed to individual task processing.

        Returns:
            list[Episode]: List of completed episodes from all tasks.
        """
        if self.workflow_queue is None:
            await self.initialize_pool()

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        # counting the number of occurences for each task_id, which gives us the rollout index
        task_id_counter = defaultdict(int)
        # pre-allocate results
        results = [None] * len(tasks)

        futures = []
        for idx, (task, task_id) in enumerate(zip(tasks, task_ids, strict=True)):
            rollout_idx = task_id_counter[task_id]
            futures.append(self.process_task_with_retry(task, task_id, rollout_idx, idx, **kwargs))
            task_id_counter[task_id] += 1

        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):  # the completion order might not be ordered
                task_id, rollout_idx, idx, episode = await future
                results[idx] = episode
                pbar.update(1)

        ordered_results: list[Episode] = results  # type: ignore[assignment]
        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                logger.info(f"Logging {len(ordered_results)} episodes to step={self.current_step}, mode={self.current_mode}, epoch={self.current_epoch}")
                self.episode_logger.log_episodes_batch(ordered_results, self.current_step, self.current_mode, self.current_epoch)
            except Exception as e:
                logger.error(f"Failed to log episodes: {e}")
                import traceback

                traceback.print_exc()

        return ordered_results

    # TODO(listar2000): eventually the agent_workflow_engine should be backend agnostic.
    async def execute_tasks_verl(self, batch: DataProto, **kwargs) -> list[Episode]:
        """Execute tasks from a Verl DataProto batch and return results.

        Args:
            batch: Verl DataProto containing tasks and metadata.
            **kwargs: Additional arguments passed to execute_tasks.

        Returns:
            DataProto: Transformed results compatible with Verl training.
        """
        assert isinstance(self.rollout_engine, VerlEngine), "Rollout engine must be a VerlEngine to invoke execute_tasks_verl"
        await self.rollout_engine.wake_up()

        is_validation = batch.meta_info.get("validate", False)
        if is_validation:
            self.rollout_engine.validate = True
            self.current_mode = "val"
        else:
            self.current_mode = "train"
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        episodes = await self.execute_tasks(tasks, task_ids, **kwargs)  # list of Episodes
        # handle data sources in the input dataproto
        if "data_source" in batch.non_tensor_batch:
            data_sources = batch.non_tensor_batch["data_source"].tolist()
            for episode, data_source in zip(episodes, data_sources, strict=True):
                episode.info["data_source"] = data_source

        self.rollout_engine.validate = False

        await self.rollout_engine.sleep()

        self.current_mode = "train"
        return episodes

    def shutdown(self):
        """Shutdown the workflow engine and cleanup resources."""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
