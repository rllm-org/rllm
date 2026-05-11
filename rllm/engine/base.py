"""Common base for flow engines.

A flow engine takes a list of tasks (raw dicts or :class:`Task` objects) and
returns a list of training-ready ``Episode`` objects.  The concrete engines —
``AgentFlowEngine`` (``rllm.engine.agentflow_engine``), ``WorkflowEngine`` and
``RemoteAgentFlowEngine`` (under ``rllm.experimental.engine``) — differ in
*how* they execute a single task.

This module factors that orchestration into ``FlowEngine`` so subclasses only
implement ``process_task_with_retry`` (and optionally the pre/post hooks).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from rllm.types import Episode

if TYPE_CHECKING:
    from rllm.utils.episode_logger import EpisodeLogger

logger = logging.getLogger(__name__)


class FlowEngine(ABC):
    """Shared scaffolding for engines that turn ``list[dict]`` tasks into ``list[Episode]``.

    Subclasses must implement :meth:`process_task_with_retry`.  They may also
    override :meth:`_pre_execute` / :meth:`_post_execute` for engine-specific
    setup or teardown around a batch.
    """

    def __init__(
        self,
        n_parallel_tasks: int = 128,
        episode_logger: EpisodeLogger | None = None,
    ) -> None:
        self.n_parallel_tasks = n_parallel_tasks
        self.episode_logger = episode_logger

        # Subclasses that need a thread pool can populate this; the default ``shutdown`` will close it.
        self.executor: ThreadPoolExecutor | None = None

        # Training step tracking, set by :meth:`set_training_step`.
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"  # "train" or "val"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0) -> None:
        """Record the training step / mode / epoch used when logging episodes."""
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    async def execute_tasks(
        self,
        tasks: list[Any],
        task_ids: list[str] | None = None,
        is_validation: bool = False,
        **kwargs,
    ) -> list[Episode]:
        """Run all tasks concurrently and return episodes in input order.

        Args:
            tasks: Tasks to process.  Subclasses decide what each element is:
                workflow/remote engines pass plain ``dict`` task specs;
                ``AgentFlowEngine`` accepts either ``dict`` or :class:`Task`.
            task_ids: Optional parallel list of task IDs; UUIDs are generated
                when omitted.  Repeated IDs become rollout indices for the same
                task (i.e. multiple rollouts of the same prompt).
            is_validation: Forwarded to ``process_task_with_retry`` and the
                pre/post hooks.
            **kwargs: Forwarded to ``process_task_with_retry``.
        """
        await self._pre_execute(is_validation=is_validation, **kwargs)

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_id_counter: dict[str, int] = defaultdict(int)
        results: list[Episode | None] = [None] * len(tasks)

        futures = []
        for idx, (task, task_id) in enumerate(zip(tasks, task_ids, strict=True)):
            rollout_idx = task_id_counter[task_id]
            task_id_counter[task_id] += 1
            futures.append(
                self.process_task_with_retry(
                    task,
                    task_id,
                    rollout_idx,
                    idx,
                    is_validation=is_validation,
                    **kwargs,
                )
            )

        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                _task_id, _rollout_idx, idx, episode = await future
                results[idx] = episode
                pbar.update(1)

        await self._post_execute(is_validation=is_validation, **kwargs)

        ordered_results: list[Episode] = results  # type: ignore[assignment]

        if self.episode_logger is not None:
            try:
                logger.info(
                    "Logging %d episodes to step=%d, mode=%s, epoch=%d",
                    len(ordered_results),
                    self.current_step,
                    self.current_mode,
                    self.current_epoch,
                )
                self.episode_logger.log_episodes_batch(
                    ordered_results,
                    self.current_step,
                    self.current_mode,
                    self.current_epoch,
                )
            except Exception as e:
                logger.error("Failed to log episodes: %s", e)
                import traceback

                traceback.print_exc()

        return ordered_results

    def shutdown(self) -> None:
        """Default shutdown: close the executor if the subclass set one."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    async def _pre_execute(self, *, is_validation: bool = False, **kwargs) -> None:
        """Run before the per-task fan-out.  Override for engine-specific setup."""

    @abstractmethod
    async def _post_execute(self, *, is_validation: bool = False, **kwargs) -> None:
        """Run after every task has completed but before episodes are logged."""

    @abstractmethod
    async def process_task_with_retry(
        self,
        task: Any,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        *,
        is_validation: bool = False,
        **kwargs,
    ) -> tuple[str, int, int, Episode]:
        """Execute a single task (with whatever retry policy the engine wants).

        Must return ``(task_id, rollout_idx, result_idx, episode)``.  The base
        ``execute_tasks`` uses ``result_idx`` to place the episode back in the
        original input order.
        """

    # ------------------------------------------------------------------
    # Helpers shared across subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def format_reward_strs(episode: Episode) -> list[str]:
        """Build human-readable per-trajectory reward strings for log lines.

        Falls back to the last step's reward if the trajectory itself has none.
        """
        reward_strs: list[str] = []
        for traj in episode.trajectories:
            reward = "N/A"
            if traj.reward is not None:
                reward = f"{traj.reward:.1f}"
            elif len(traj.steps) > 0:
                reward = f"{traj.steps[-1].reward:.1f}"
            reward_strs.append(f"{traj.name}: {reward}")
        return reward_strs
