from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from tqdm import tqdm

from rllm.agents.agent import Episode
from rllm.engine.rollout import RolloutEngine
from rllm.misc import colorful_print
from rllm.workflows.workflow import TerminationReason, Workflow

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger(__name__)


class AgentWorkflowEngine:
    def __init__(self, workflow_cls: type[Workflow], workflow_args: dict, rollout_engine: RolloutEngine, config=None, n_parallel_tasks: int = 128, retry_limit: int = 3, raise_on_error: bool = True, episode_logger=None, **kwargs):
        """Initialize the AgentWorkflowEngine.

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
        if self.workflow_queue is not None:
            return
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for i in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(rollout_engine=self.rollout_engine, executor=self.executor, **self.workflow_args)
            assert workflow.is_multithread_safe(), "Workflows must contain only thread-save environments"
            self.workflow_queue.put_nowait(workflow)

    async def process_task_with_retry(self, task: dict, task_id: str, rollout_idx: int, **kwargs) -> tuple[str, int, Episode]:
        """Process a single task rollout with retry logic based on termination reasons.

        Args:
            task: Task dictionary containing the task specification.
            task_id: Unique identifier for the task.
            rollout_idx: Index of this rollout attempt for the task.
            **kwargs: Additional arguments passed to the workflow.

        Returns:
            tuple[str, int, Episode]: Task ID, rollout index, and completed episode.

        Raises:
            Exception: If task fails permanently after retry_limit attempts and raise_on_error is True.
        """
        workflow = await self.workflow_queue.get()
        try:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                episode = await workflow.run_with_termination_handling(task=task, uid=uid, **kwargs)

                # Display rewards for all trajectories
                rewards_str = ", ".join([f"{traj.name}: {traj.reward:.1f}" if traj.reward is not None else f"{traj.name}: none" for traj in episode.trajectories])
                colorful_print(f"[{uid}] Rollout completed. Rewards: {rewards_str}, Termination: {episode.termination_reason}", fg="green" if episode.is_correct else "yellow")

                if episode.termination_reason != TerminationReason.ERROR:
                    return task_id, rollout_idx, episode

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

        task_states = defaultdict(lambda: {"idx": None, "task": None, "episodes": [], "completed": 0, "total_rollouts": 0, "is_complete": False})

        futures = []
        idx_counter = 0
        for task, task_id in zip(tasks, task_ids, strict=True):
            state = task_states[task_id]
            if state["idx"] is None:  # First time seeing this task_id
                state["idx"] = idx_counter
                state["task"] = task
                idx_counter += 1
            rollout_idx = state["total_rollouts"]
            futures.append(self.process_task_with_retry(task, task_id, rollout_idx, **kwargs))
            state["total_rollouts"] += 1

        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, episode = await future

                state = task_states[task_id]
                state["episodes"].append(episode)
                state["completed"] += 1
                pbar.update(1)

        results = []
        sorted_tasks = sorted(task_states.keys(), key=lambda task_id: task_states[task_id]["idx"])
        for task_id in sorted_tasks:
            results.extend(task_states[task_id]["episodes"])

        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                logger.info(f"Logging {len(results)} episodes to step={self.current_step}, mode={self.current_mode}, epoch={self.current_epoch}")
                self.episode_logger.log_episodes_batch(results, self.current_step, self.current_mode, self.current_epoch)
            except Exception as e:
                logger.error(f"Failed to log episodes: {e}")
                import traceback

                traceback.print_exc()

        return results

    async def execute_tasks_verl(self, batch: DataProto, **kwargs) -> list[Episode]:
        """Execute tasks from a Verl DataProto batch and return results.

        Args:
            batch: Verl DataProto containing tasks and metadata.
            **kwargs: Additional arguments passed to execute_tasks.

        Returns:
            DataProto: Transformed results compatible with Verl training.
        """
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
        self.rollout_engine.validate = False

        await self.rollout_engine.sleep()

        self.current_mode = "train"
        return episodes

    def shutdown(self):
        """Shutdown the workflow engine and cleanup resources."""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
