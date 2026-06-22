"""TrajectoryGroupBuffer for async training.

Accumulates episodes, processes into ready-to-train trajectory groups,
with optional NVMe offloading for memory management.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm

from rllm.trainer.algorithms import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
    collect_reward_and_advantage_from_trajectory_groups,
)
from rllm.trainer.algorithms.transform import transform_episodes_to_trajectory_groups
from rllm.trainer.metrics_aggregator import MetricsAggregator
from rllm.trainer.sync_coordinator import SyncCoordinator
from rllm.types import Episode, TrajectoryGroup
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)


@dataclass
class TaskBatch:
    """All trajectory groups produced from one task's episodes, plus stripped episodes for UI logging."""

    groups: list[TrajectoryGroup]
    episodes: list[Episode] = field(default_factory=list)


class TrajectoryGroupBuffer:
    """Accumulates episodes, processes into trajectory groups, yields to training.

    When all rollouts for a task arrive:
    1. Record episode-level metrics to aggregator (before any filtering)
    2. Transform episodes -> trajectory groups
    3. Compact filtering + drop groups with < min_trajs_per_group
    4. Compute advantages
    5. If rejection sampling enabled: drop groups with all-zero advantage
    6. Queue the task batch for training

    Filtered groups are reported directly to the coordinator (which tracks
    throttle slots and filter counts). Only non-empty task batches are queued.
    All metrics flow through the shared MetricsAggregator.

    Optionally offloads pending episodes and/or queued task batches to
    disk to reduce memory pressure (disabled by default).
    """

    def __init__(
        self,
        group_size: int,
        coordinator: SyncCoordinator,
        aggregator: MetricsAggregator,
        algorithm_config: AlgorithmConfig,
        transform_config: TransformConfig,
        cf_config: CompactFilteringConfig,
        rs_config: RejectionSamplingConfig,
        episode_offload_dir: str | None = None,
        trajectory_group_offload_dir: str | None = None,
        pbar: tqdm | None = None,
    ):
        self._group_size = group_size
        self._coordinator = coordinator
        self._aggregator = aggregator
        self._algorithm_config = algorithm_config
        self._transform_config = transform_config
        self._cf_config = cf_config
        self._rs_config = rs_config
        self._pbar = pbar

        # Episode offloading: pending episodes serialized to disk
        self._episode_offload_dir = episode_offload_dir
        if episode_offload_dir:
            os.makedirs(episode_offload_dir, exist_ok=True)
        self._pending: dict[str, list[Episode | str]] = {}  # str = offloaded file path

        # Trajectory group offloading: queued task batches serialized to disk
        self._tg_offload_dir = trajectory_group_offload_dir
        if trajectory_group_offload_dir:
            os.makedirs(trajectory_group_offload_dir, exist_ok=True)
        self._queue: asyncio.Queue[TaskBatch | str | None] = asyncio.Queue()
        self._training_queue_size = 0
        self._filtered_count = 0
        self._consumed_count = 0
        self._training_step = 0
        self._queue_update_event = asyncio.Event()
        self._generation_complete = False

    def set_training_step(self, step: int) -> None:
        self._training_step = step
        self._refresh_pbar_counters()

    def _refresh_pbar_counters(self) -> None:
        if self._pbar is not None:
            self._pbar.set_postfix(
                step=self._training_step,
                queued=self._training_queue_size,
                filtered=self._filtered_count,
                consumed=self._consumed_count,
                refresh=False,
            )

    def _record_classified_prompt_group(self) -> None:
        self._refresh_pbar_counters()
        if self._pbar is not None:
            self._pbar.update(1)

    async def _offload_episode(self, task_id: str, episode: Episode) -> str:
        """Serialize episode to disk, return file path."""
        idx = len(self._pending.get(task_id, []))
        path = os.path.join(self._episode_offload_dir, f"{task_id}_{idx}.pkl")
        await asyncio.to_thread(self._pickle_dump, path, episode)
        return path

    async def _load_pending_episodes(self, task_id: str) -> list[Episode]:
        """Load all pending episodes for a task, deserializing offloaded ones."""
        episodes = []
        for item in self._pending.pop(task_id, []):
            if isinstance(item, str):
                ep = await asyncio.to_thread(self._pickle_load, item)
                episodes.append(ep)
            else:
                episodes.append(item)
        return episodes

    async def _offload_task_batch(self, batch: TaskBatch) -> str:
        """Serialize task batch to disk, return file path."""
        fd, path = tempfile.mkstemp(dir=self._tg_offload_dir, suffix=".pkl")
        os.close(fd)
        await asyncio.to_thread(self._pickle_dump, path, batch)
        return path

    async def _load_task_batch(self, item: TaskBatch | str) -> TaskBatch:
        """Load task batch, deserializing if offloaded."""
        if isinstance(item, str):
            return await asyncio.to_thread(self._pickle_load, item)
        return item

    @staticmethod
    def _pickle_dump(path: str, obj) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _pickle_load(path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        os.remove(path)
        return obj

    async def add_episode(self, task_id: str, episode: Episode) -> bool:
        """Add episode. When group completes, process and queue task batch."""
        if self._generation_complete:
            logger.warning("Ignoring episode for task %s after generation was marked complete", task_id)
            return False

        # Offload episode to disk if enabled
        if self._episode_offload_dir:
            path = await self._offload_episode(task_id, episode)
            self._pending.setdefault(task_id, []).append(path)
        else:
            self._pending.setdefault(task_id, []).append(episode)

        if len(self._pending[task_id]) < self._group_size:
            return False

        # Load all episodes
        if self._episode_offload_dir:
            episodes = await self._load_pending_episodes(task_id)
        else:
            episodes = self._pending.pop(task_id, [])

        weight_version = self._min_weight_version(episodes)

        # 1. Record episode-level metrics (includes filtered tasks)
        self._record_episode_metrics(episodes)

        # 2. Transform episodes -> trajectory groups
        traj_groups, transform_metrics = transform_episodes_to_trajectory_groups(
            episodes,
            self._transform_config,
            self._cf_config,
        )
        self._aggregator.record_dict(transform_metrics)

        # 3. Drop groups with too few trajectories
        before_min_traj = len(traj_groups)
        traj_groups = [g for g in traj_groups if len(g.trajectories) >= self._rs_config.min_trajs_per_group]
        self._aggregator.record("groups/dropped_min_trajs", before_min_traj - len(traj_groups))

        if not traj_groups:
            if before_min_traj > 0:
                filter_reason = "min_trajs"
            elif self._all_episodes_compact_filtered(episodes):
                filter_reason = "compact_filtering"
            else:
                filter_reason = "no_trajectory_groups"
            self._log_prompt_group_finished(
                task_id=task_id,
                episodes=episodes,
                status="filtered",
                reason=filter_reason,
                groups_after_transform=before_min_traj,
                groups_after_min_trajs=0,
                groups_after_reward_filter=0,
            )
            self._coordinator.on_group_filtered()
            self._filtered_count += 1
            self._record_classified_prompt_group()
            return True

        # 4. Compute advantages
        adv_metrics = collect_reward_and_advantage_from_trajectory_groups(
            traj_groups,
            self._algorithm_config,
        )
        self._aggregator.record_dict(adv_metrics)

        # 5. Rejection sampling: drop groups with all-zero advantage
        filtered_zero_adv = 0
        if self._rs_config.filter_uniform_groups:
            before_adv = len(traj_groups)
            traj_groups = [g for g in traj_groups if any(abs(step.advantage) > 1e-8 for traj in g.trajectories for step in traj.steps if step.advantage is not None)]
            filtered_zero_adv = before_adv - len(traj_groups)
        self._aggregator.record("groups/dropped_zero_adv", filtered_zero_adv)

        if not traj_groups:
            self._log_prompt_group_finished(
                task_id=task_id,
                episodes=episodes,
                status="filtered",
                reason="uniform_reward",
                groups_after_transform=before_min_traj,
                groups_after_min_trajs=before_adv,
                groups_after_reward_filter=0,
            )
            self._coordinator.on_group_filtered()
            self._filtered_count += 1
            self._record_classified_prompt_group()
            return True

        # 6. Set weight version and queue
        for g in traj_groups:
            g.weight_version = weight_version

        batch = TaskBatch(groups=traj_groups, episodes=episodes)
        if self._tg_offload_dir:
            await self._queue.put(await self._offload_task_batch(batch))
        else:
            await self._queue.put(batch)
        self._training_queue_size += 1
        self._queue_update_event.set()
        self._record_classified_prompt_group()

        # Per-task fine-grained accounting: a task is consumed as one GRPO group,
        # which collapses into training rows/datums under prefix-merge. Surface
        # trajectories / steps (turns) / datums (rows after merge) so the merge
        # ratio is visible per task, not just as a batch aggregate.
        n_traj = sum(len(g.trajectories) for g in traj_groups)
        n_steps = sum(len(t.steps) for g in traj_groups for t in g.trajectories)
        n_datums = sum(self._segment_count(t) for g in traj_groups for t in g.trajectories)
        logger.info(
            "Task %s queued: %d group(s), %d trajectories, %d steps -> %d datums (rows) [%.2f steps/datum]",
            task_id, len(traj_groups), n_traj, n_steps, n_datums,
            (n_steps / n_datums) if n_datums else 0.0,
        )

        self._log_prompt_group_finished(
            task_id=task_id,
            episodes=episodes,
            status="queued",
            reason="accepted",
            groups_after_transform=before_min_traj,
            groups_after_min_trajs=len(traj_groups) + filtered_zero_adv,
            groups_after_reward_filter=len(traj_groups),
        )

        return True

    async def get(self) -> TaskBatch | None:
        """Get next task batch. Returns None when generation is done and buffer is drained."""
        item = await self._queue.get()
        if item is None:
            return None
        self._training_queue_size = max(0, self._training_queue_size - 1)
        self._consumed_count += 1
        self._refresh_pbar_counters()
        return await self._load_task_batch(item)

    async def get_many(self, count: int) -> list[TaskBatch] | None:
        """Get a full forward/backward chunk, or None if generation ended first."""
        while self._training_queue_size < count:
            if self._generation_complete:
                return None
            self._queue_update_event.clear()
            if self._training_queue_size >= count or self._generation_complete:
                continue
            await self._queue_update_event.wait()

        items = []
        for _ in range(count):
            item = await self._queue.get()
            if item is None:
                return None
            items.append(await self._load_task_batch(item))

        self._training_queue_size = max(0, self._training_queue_size - count)
        self._consumed_count += count
        self._refresh_pbar_counters()
        return items

    def mark_generation_complete(self) -> None:
        """Signal that generation is finished. Flushes incomplete groups and enqueues a sentinel."""
        if self._generation_complete:
            return
        self._generation_complete = True
        for task_id in list(self._pending.keys()):
            items = self._pending.pop(task_id, [])
            for item in items:
                if isinstance(item, str):
                    try:
                        os.remove(item)
                    except OSError:
                        pass
            self._coordinator.on_group_filtered()
            self._filtered_count += 1
            self._record_classified_prompt_group()
        self._queue.put_nowait(None)
        self._queue_update_event.set()

    def stats(self) -> dict:
        return {
            "async/buffer_qsize": self._training_queue_size,
            "async/buffer_pending": len(self._pending),
            "async/buffer_filtered": self._filtered_count,
            "async/buffer_consumed": self._consumed_count,
        }

    def _record_episode_metrics(self, episodes: list[Episode]) -> None:
        """Record episode-level metrics to aggregator (all episodes, including filtered)."""
        for ep in episodes:
            reason = ep.termination_reason or TerminationReason.UNKNOWN
            for r in TerminationReason:
                self._aggregator.record(
                    f"episode/termination_reason/{r.value}",
                    1.0 if reason == r else 0.0,
                )
            for k, v in ep.metrics.items():
                try:
                    self._aggregator.record(f"episode/{k}", float(v))
                except (TypeError, ValueError):
                    continue

            # Episode-level totals across all trajectories
            total_turns = sum(len(traj.steps) for traj in ep.trajectories)
            total_prompt_tokens = sum(len(s.prompt_ids) for traj in ep.trajectories for s in traj.steps)
            total_response_tokens = sum(len(s.response_ids) for traj in ep.trajectories for s in traj.steps)
            self._aggregator.record("episode/num_turns", total_turns)
            self._aggregator.record("episode/prompt_tokens", total_prompt_tokens)
            self._aggregator.record("episode/response_tokens", total_response_tokens)
            self._aggregator.record("episode/correct", 1.0 if ep.is_correct else 0.0)

    def _all_episodes_compact_filtered(self, episodes: list[Episode]) -> bool:
        return all(self._cf_config.should_mask(ep.termination_reason or TerminationReason.UNKNOWN) for ep in episodes)

    @staticmethod
    def _termination_value(reason: TerminationReason | str) -> str:
        return str(getattr(reason, "value", reason))

    @staticmethod
    def _segment_count(traj) -> int:
        """Number of training rows/datums a trajectory becomes under prefix-merge.

        Each step whose ``prompt_ids`` is NOT a byte-prefix-extension of the
        running cumulative sequence starts a new row — mirroring the backend
        transform's datum split. =1 for a healthy cumulative trajectory (all
        turns merge into one row); >1 when the prefix chain breaks.
        """
        full: list[int] | None = None
        segments = 0
        for step in traj.steps:
            ids = list(step.prompt_ids)
            if full is None or ids[: len(full)] != full:
                segments += 1
            full = ids + list(step.response_ids)
        return max(segments, 1)

    def _log_prompt_group_finished(
        self,
        *,
        task_id: str,
        episodes: list[Episode],
        status: str,
        reason: str,
        groups_after_transform: int,
        groups_after_min_trajs: int,
        groups_after_reward_filter: int,
    ) -> None:
        termination_counts = Counter(self._termination_value(ep.termination_reason or TerminationReason.UNKNOWN) for ep in episodes)
        compact_masked = Counter(
            self._termination_value(ep.termination_reason or TerminationReason.UNKNOWN) for ep in episodes if self._cf_config.should_mask(ep.termination_reason or TerminationReason.UNKNOWN)
        )
        rewards = []
        for ep in episodes:
            reward = None
            for traj in ep.trajectories:
                if traj.reward is not None:
                    reward = traj.reward
                elif traj.steps:
                    reward = traj.steps[-1].reward
            rewards.append(reward)

        logger.debug(
            "Prompt group finished task_id=%s status=%s reason=%s episodes=%d rewards=%s "
            "terminations=%s compact_masked=%s groups_after_transform=%d "
            "groups_after_min_trajs=%d groups_after_reward_filter=%d",
            task_id,
            status,
            reason,
            len(episodes),
            rewards,
            dict(termination_counts),
            dict(compact_masked),
            groups_after_transform,
            groups_after_min_trajs,
            groups_after_reward_filter,
        )

    @staticmethod
    def _min_weight_version(episodes: list[Episode]) -> int:
        min_v = float("inf")
        for ep in episodes:
            for traj in ep.trajectories:
                for step in traj.steps:
                    if step.weight_version is not None:
                        min_v = min(min_v, step.weight_version)
        return int(min_v) if min_v != float("inf") else 0
