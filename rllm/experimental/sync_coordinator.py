"""SyncCoordinator: manages rollout quotas and parameter sync timing for fully-async training."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class SyncCoordinatorConfig:
    mini_batch_size: int  # episode groups per optimizer step
    group_size: int  # episodes per group (rollout.n)
    staleness_threshold: float
    trigger_parameter_sync_step: int

    @property
    def max_rollout_quota(self) -> int:
        """Max outstanding groups (dispatched but not yet consumed by training)."""
        return int((1 + self.staleness_threshold) * self.trigger_parameter_sync_step * self.mini_batch_size)


class SyncCoordinator:
    """Coordinates rollout scheduling and parameter sync between generation and training loops."""

    def __init__(self, config: SyncCoordinatorConfig):
        self.config = config

        self._policy_version: int = 0
        self._outstanding: int = 0  # groups dispatched but not yet consumed by training
        self._steps_since_sync: int = 0
        self._total_syncs: int = 0
        self._total_groups_filtered: int = 0

        # Throttle — blocks generation when outstanding >= max_rollout_quota
        self._throttle_event: asyncio.Event = asyncio.Event()
        self._throttle_event.set()

        # Generation pause — blocks generation during validation or weight sync
        self._generation_paused: asyncio.Event = asyncio.Event()
        self._generation_paused.set()

        self.generation_done: bool = False

    @property
    def policy_version(self) -> int:
        return self._policy_version

    # --- Throttle ---

    def on_group_dispatched(self) -> None:
        """Generation loop dispatched one prompt (n rollouts)."""
        self._outstanding += 1
        if self._outstanding >= self.config.max_rollout_quota:
            self._throttle_event.clear()

    def on_group_consumed(self) -> None:
        """Training loop consumed one group from the buffer."""
        self._outstanding = max(0, self._outstanding - 1)
        self._throttle_event.set()

    def on_group_filtered(self) -> None:
        """Accumulator filtered out a uniform group. Frees throttle slot and tracks count."""
        self._total_groups_filtered += 1
        self.on_group_consumed()

    async def wait_for_throttle(self) -> None:
        """Generation loop blocks here when quota is full."""
        await self._throttle_event.wait()

    def has_quota(self) -> bool:
        """Whether the generation loop can dispatch another group."""
        return self._outstanding < self.config.max_rollout_quota

    # --- Weight sync ---

    def on_training_step_complete(self) -> None:
        self._steps_since_sync += 1

    def should_sync(self) -> bool:
        return self._steps_since_sync >= self.config.trigger_parameter_sync_step

    def on_sync_complete(self) -> None:
        self._policy_version += 1
        self._steps_since_sync = 0
        self._total_syncs += 1

    # --- Generation pause (for validation / non-partial weight sync) ---

    def pause_generation(self) -> None:
        self._generation_paused.clear()

    def resume_generation(self) -> None:
        self._generation_paused.set()

    async def wait_for_generation_allowed(self) -> None:
        await self._generation_paused.wait()

    def stats(self) -> dict:
        return {
            "async/policy_version": self._policy_version,
            "async/outstanding_groups": self._outstanding,
            "async/steps_since_sync": self._steps_since_sync,
            "async/max_rollout_quota": self.config.max_rollout_quota,
            "async/total_syncs": self._total_syncs,
            "async/total_groups_filtered": self._total_groups_filtered,
        }
