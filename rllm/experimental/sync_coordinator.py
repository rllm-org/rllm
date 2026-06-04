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
        self._in_flight: int = 0  # groups dispatched but not yet consumed/filtered
        self._steps_since_sync: int = 0
        self._total_syncs: int = 0

        # Throttle — blocks generation when dispatched_since_sync >= max_rollout_quota
        self._throttle_event: asyncio.Event = asyncio.Event()
        self._throttle_event.set()

        # Generation pause — blocks generation during validation or weight sync
        self._generation_paused: asyncio.Event = asyncio.Event()
        self._generation_paused.set()

        # Tracks in-flight async rollout tasks for drain/wait logic
        self._in_flight_tasks: set[asyncio.Task] = set()
        self._task_errors: list[BaseException] = []
        self._task_error_event: asyncio.Event = asyncio.Event()

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
        """Accumulator filtered out a group. Releases in-flight and quota."""
        self._in_flight = max(0, self._in_flight - 1)
        self._quota_used = max(0, self._quota_used - 1)
        if self._quota_used < self.config.max_rollout_quota:
            self._throttle_event.set()

    async def wait_for_throttle(self) -> None:
        """Generation loop blocks here when dispatch window is full."""
        await self._throttle_event.wait()
        self.raise_if_task_failed()

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
        self.raise_if_task_failed()

    # --- In-flight task tracking ---

    def track_task(self, task: asyncio.Task) -> None:
        """Register an in-flight rollout task."""
        self._in_flight_tasks.add(task)

        def _on_done(done_task: asyncio.Task) -> None:
            self._in_flight_tasks.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                self.record_task_error(exc)

        task.add_done_callback(_on_done)

    def record_task_error(self, exc: BaseException) -> None:
        """Record a rollout task failure and release waits so it can surface."""
        if not any(existing is exc for existing in self._task_errors):
            self._task_errors.append(exc)
        self._task_error_event.set()
        self._throttle_event.set()
        self._generation_paused.set()

    def raise_if_task_failed(self) -> None:
        if not self._task_errors:
            return
        first = self._task_errors[0]
        self._task_errors.clear()
        self._task_error_event.clear()
        raise RuntimeError("Async rollout task failed") from first

    async def wait_for_task_error(self) -> None:
        """Block until any tracked rollout task fails, then raise that failure."""
        await self._task_error_event.wait()
        self.raise_if_task_failed()

    def cancel_tracked_tasks(self) -> None:
        """Cancel rollout tasks that were dispatched but are no longer useful."""
        for task in list(self._in_flight_tasks):
            task.cancel()

    async def wait_for_drain(self) -> None:
        """Wait for all in-flight rollout tasks to complete."""
        while self._in_flight_tasks:
            await asyncio.sleep(0.1)
        self.raise_if_task_failed()

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
