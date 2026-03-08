"""SyncCoordinator: manages rollout quotas and parameter sync timing for fully-async training."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class SyncCoordinatorConfig:
    """Configuration derived from trainer config and async config for the coordinator to use."""

    train_batch_size: int  # from config.data.train_batch_size
    group_size: int  # from config.rllm.rollout.n
    staleness_threshold: float  # from async config
    trigger_parameter_sync_step: int  # from async config
    requeue_stale: bool  # from async config
    num_minibatches: int  # from async config

    @property
    def episodes_per_train_step(self) -> int:
        """Number of episodes per training step."""
        return self.train_batch_size * self.group_size

    @property
    def max_rollout_quota(self) -> int:
        """Max episodes in-flight + in-queue between syncs."""
        return int((1 + self.staleness_threshold) * self.trigger_parameter_sync_step * self.episodes_per_train_step)


class SyncCoordinator:
    """Coordinates rollout scheduling and parameter sync between generation and training loops.

    Core responsibility: control how many episodes can be in-flight + queued,
    and when to trigger weight synchronization.
    """

    def __init__(self, config: SyncCoordinatorConfig):
        self.config = config

        # State
        self._policy_version: int = 0
        self._scheduled_count: int = 0  # in-flight episodes not yet in buffer
        self._stale_requeue_count: int = 0  # stale tasks to re-add to quota
        self._steps_since_sync: int = 0  # training steps since last sync
        self._total_stale_discarded: int = 0

        # Events
        self.sync_complete_event: asyncio.Event = asyncio.Event()
        self.sync_complete_event.set()  # initially unblocked
        self.generation_done: bool = False

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def compute_new_schedule_count(self, remain_in_queue: int) -> int:
        """Compute how many new episodes the generation loop should schedule.

        Formula: new = max_rollout_quota - remain_in_queue - scheduled + requeue_stale
        """
        available = self.config.max_rollout_quota - remain_in_queue - self._scheduled_count + self._stale_requeue_count
        self._stale_requeue_count = 0  # consumed
        return max(0, available)

    def on_episodes_scheduled(self, count: int) -> None:
        """Called when the generation loop dispatches tasks."""
        self._scheduled_count += count

    def on_episode_generated(self, count: int = 1) -> None:
        """Called when an episode arrives in the buffer (no longer in-flight)."""
        self._scheduled_count = max(0, self._scheduled_count - count)

    def on_training_step_complete(self) -> None:
        """Called after a gradient update."""
        self._steps_since_sync += 1

    def should_sync(self) -> bool:
        """Whether it's time to synchronize parameters."""
        return self._steps_since_sync >= self.config.trigger_parameter_sync_step

    def on_sync_complete(self) -> None:
        """Called after weight sync. Bumps policy version, resets counters, signals gen loop."""
        self._policy_version += 1
        self._steps_since_sync = 0
        self.sync_complete_event.set()

    def on_stale_discarded(self, count: int, requeue: bool) -> None:
        """Called when the training loop discards stale episodes."""
        self._total_stale_discarded += count
        if requeue:
            self._stale_requeue_count += count

    def is_episode_stale(self, ep_version: int) -> bool:
        """Check if an episode is too stale to use.

        An episode is stale if the version gap exceeds:
        staleness_threshold * trigger_parameter_sync_step
        """
        if self.config.staleness_threshold == 0.0:
            # On-policy: only current version is acceptable
            return ep_version < self._policy_version
        max_gap = self.config.staleness_threshold * self.config.trigger_parameter_sync_step
        return (self._policy_version - ep_version) > max_gap

    def stats(self) -> dict:
        return {
            "async/policy_version": self._policy_version,
            "async/scheduled_count": self._scheduled_count,
            "async/steps_since_sync": self._steps_since_sync,
            "async/total_stale_discarded": self._total_stale_discarded,
            "async/max_rollout_quota": self.config.max_rollout_quota,
        }
