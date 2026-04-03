"""SyncCoordinator: manages rollout quotas and parameter sync timing for fully-async training."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class SyncCoordinatorConfig:
    mini_batch_size: int     # episode groups per optimizer step
    group_size: int          # episodes per group (rollout.n)
    staleness_threshold: float
    trigger_parameter_sync_step: int

    @property
    def max_rollout_quota(self) -> int:
        """Max dispatches per sync window (Verl/AReaL formulation)."""
        return int((1 + self.staleness_threshold) * self.trigger_parameter_sync_step * self.mini_batch_size)


class SyncCoordinator:
    """Coordinates rollout scheduling and parameter sync between generation and training loops.

    Uses a per-sync-window dispatch counter (matching Verl/AReaL). The counter
    resets only on weight sync, not on consume. This guarantees zero staleness
    when staleness_threshold=0.
    """

    def __init__(self, config: SyncCoordinatorConfig):
        self.config = config

        self._weight_version: int = 0
        self._quota_used: int = 0  # groups counting toward current sync window quota (includes carryover)
        self._in_flight: int = 0              # groups dispatched but not yet consumed/filtered
        self._steps_since_sync: int = 0
        self._total_syncs: int = 0

        # Throttle — blocks generation when dispatched_since_sync >= max_rollout_quota
        self._throttle_event: asyncio.Event = asyncio.Event()
        self._throttle_event.set()

        # Generation pause — blocks generation during validation or weight sync
        self._generation_paused: asyncio.Event = asyncio.Event()
        self._generation_paused.set()

    @property
    def weight_version(self) -> int:
        return self._weight_version

    # --- Throttle ---

    def on_group_dispatched(self) -> None:
        """Generation loop dispatched one prompt (n rollouts)."""
        self._quota_used += 1
        self._in_flight += 1
        if self._quota_used >= self.config.max_rollout_quota:
            self._throttle_event.clear()

    def on_group_consumed(self) -> None:
        """Training loop consumed one group from the buffer."""
        self._in_flight = max(0, self._in_flight - 1)

    def on_group_filtered(self) -> None:
        """Accumulator filtered out a group. Decrements in-flight count."""
        self._in_flight = max(0, self._in_flight - 1)

    async def wait_for_throttle(self) -> None:
        """Generation loop blocks here when dispatch window is full."""
        await self._throttle_event.wait()

    def has_quota(self) -> bool:
        """Whether the generation loop can dispatch another group."""
        return self._quota_used < self.config.max_rollout_quota

    # --- Weight sync ---

    def on_training_step_complete(self) -> None:
        self._steps_since_sync += 1

    def should_sync(self) -> bool:
        return self._steps_since_sync >= self.config.trigger_parameter_sync_step

    def on_sync_complete(self) -> None:
        self._weight_version += 1
        self._steps_since_sync = 0
        self._total_syncs += 1
        # Reset dispatch window. In-flight items span the sync boundary —
        # they were dispatched with old weights and count toward the new window.
        self._quota_used = self._in_flight
        if self._quota_used < self.config.max_rollout_quota:
            self._throttle_event.set()

    # --- Generation pause (for validation / weight sync if partial_rollout is False) ---

    def pause_generation(self) -> None:
        self._generation_paused.clear()

    def resume_generation(self) -> None:
        self._generation_paused.set()

    async def wait_for_generation_allowed(self) -> None:
        await self._generation_paused.wait()

    def stats(self) -> dict:
        return {
            "async/weight_version": self._weight_version,
            "async/dispatched_since_sync": self._quota_used - self._in_flight,
            "async/quota_used": self._quota_used,
            "async/in_flight_groups": self._in_flight,
            "async/steps_since_sync": self._steps_since_sync,
            "async/max_rollout_quota": self.config.max_rollout_quota,
            "async/total_syncs": self._total_syncs,
        }
