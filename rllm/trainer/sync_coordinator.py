"""SyncCoordinator: manages rollout capacity and parameter sync timing for fully-async training."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class SyncCoordinatorConfig:
    mini_batch_size: int  # episode groups per optimizer step
    group_size: int  # episodes per group (rollout.n)
    staleness_threshold: float
    trigger_parameter_sync_step: int
    max_concurrent_rollouts: int  # rollout tasks kept in flight (== workflow.n_parallel_tasks)

    @property
    def max_in_flight_groups(self) -> int:
        """Staleness cap: groups dispatched-but-not-consumed may not exceed this.

        Keeps generation at most (1 + staleness_threshold) mini-batches ahead of
        training. Equivalent to the version-scaled ceiling
        ``(staleness + version + 1) * mini_batch``, just expressed in the
        consumption frame; the two match because the version advances one
        mini-batch per optimizer step.
        """
        return int((1 + self.staleness_threshold) * self.trigger_parameter_sync_step * self.mini_batch_size)


class SyncCoordinator:
    """Coordinates rollout scheduling and parameter sync between generation and training loops.

    Dispatch is governed by a continuously-evaluated capacity::

        capacity > 0  iff  in_flight_groups < max_in_flight_groups        (staleness)
                      and  running_rollouts  < max_concurrent_rollouts    (concurrency)

    Both terms are re-checked (and generation is woken) whenever a group is
    consumed/filtered or a rollout task completes -- so freed budget is refilled
    immediately rather than in a burst at weight sync. This keeps the rollout
    workers saturated across a step instead of front-loading the sync window and
    idling until the next sync.
    """

    def __init__(self, config: SyncCoordinatorConfig):
        self.config = config

        self._weight_version: int = 0
        self._in_flight: int = 0  # groups dispatched but not yet consumed/filtered (staleness budget)
        self._steps_since_sync: int = 0
        self._total_syncs: int = 0

        # Capacity gate — set whenever staleness or concurrency budget frees up.
        # Generation waits on it and re-checks has_capacity() (level-triggered).
        self._capacity_event: asyncio.Event = asyncio.Event()
        self._capacity_event.set()

        # Generation pause — blocks generation during validation or weight sync
        self._generation_paused: asyncio.Event = asyncio.Event()
        self._generation_paused.set()

        # Tracks in-flight async rollout tasks for drain/wait logic + concurrency term
        self._in_flight_tasks: set[asyncio.Task] = set()
        self._task_errors: list[BaseException] = []
        self._task_error_event: asyncio.Event = asyncio.Event()

    @property
    def weight_version(self) -> int:
        return self._weight_version

    # --- Capacity (continuous dispatch control) ---

    def _signal_capacity(self) -> None:
        """Wake the generation loop to re-evaluate capacity."""
        self._capacity_event.set()

    def has_capacity(self) -> bool:
        """Whether the generation loop may dispatch another group right now.

        min(concurrency_capacity, staleness_capacity) > 0.
        """
        staleness_ok = self._in_flight < self.config.max_in_flight_groups
        concurrency_ok = len(self._in_flight_tasks) < self.config.max_concurrent_rollouts
        return staleness_ok and concurrency_ok

    async def wait_for_capacity(self) -> None:
        """Block until there is capacity to dispatch another group.

        Level-triggered: signallers may set the event when only one of the two
        constraints frees, so we loop and re-check has_capacity().
        """
        while not self.has_capacity():
            self._capacity_event.clear()
            await self._capacity_event.wait()
            self.raise_if_task_failed()
        self.raise_if_task_failed()

    def on_group_dispatched(self) -> None:
        """Generation loop dispatched one group (group_size rollouts)."""
        self._in_flight += 1

    def on_group_consumed(self) -> None:
        """Training loop consumed one group from the buffer. Frees staleness budget."""
        self._in_flight = max(0, self._in_flight - 1)
        self._signal_capacity()

    def on_group_filtered(self) -> None:
        """Accumulator filtered out a group. Frees staleness budget."""
        self._in_flight = max(0, self._in_flight - 1)
        self._signal_capacity()

    # --- Weight sync ---

    def on_training_step_complete(self) -> None:
        self._steps_since_sync += 1

    def should_sync(self) -> bool:
        return self._steps_since_sync >= self.config.trigger_parameter_sync_step

    def on_sync_complete(self) -> None:
        self._weight_version += 1
        self._steps_since_sync = 0
        self._total_syncs += 1

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
            # A completed rollout frees a concurrency slot — refill immediately.
            self._signal_capacity()
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
        self._capacity_event.set()
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
            "async/in_flight_groups": self._in_flight,
            "async/running_rollouts": len(self._in_flight_tasks),
            "async/max_in_flight_groups": self.config.max_in_flight_groups,
            "async/max_concurrent_rollouts": self.config.max_concurrent_rollouts,
            "async/steps_since_sync": self._steps_since_sync,
            "async/total_syncs": self._total_syncs,
        }
