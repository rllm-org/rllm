"""A per-run warm queue that prefetches sandboxes ahead of consumption.

The queue walks a run's ordered task schedule with a few background threads,
calling :func:`rllm.sandbox.snapshot.get_sandbox` (Phase 1) for each upcoming
task and parking the result keyed by ``env_key``. A consumer thread (the
:class:`rllm.hooks.SandboxTaskHooks` setup running in the engine's executor)
then :meth:`pop`\\s a ready sandbox for its task instead of creating one
inline, overlapping creation with rollout. It is snapshot-agnostic: a snapshot
hit only makes a fill faster; the consumer pops a ready sandbox either way.

``size`` bounds how many sandboxes are warm (ready + in flight) at once, so the
queue stays a fixed distance ahead of the consumption frontier rather than
pre-creating the whole dataset. There is no reuse: :meth:`pop` hands ownership
to the consumer, which closes the sandbox; the queue only closes the prefetched
tail the run never consumed.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from rllm.sandbox.snapshot import env_key_for, get_sandbox

if TYPE_CHECKING:
    from rllm.sandbox.protocol import Sandbox
    from rllm.sandbox.snapshot import SnapshotRegistry
    from rllm.types import Task

logger = logging.getLogger(__name__)


def _close(sandbox: Sandbox) -> None:
    try:
        sandbox.close()
    except Exception:
        logger.exception("warm queue: sandbox close failed")


class WarmQueue:
    """Prefetch buffer of ready sandboxes for a run's ordered schedule.

    All shared state lives under one :class:`threading.Condition`. A *slot* is
    held from the moment a filler claims a schedule item, through its
    ``get_sandbox`` call, while the sandbox sits ready, until :meth:`pop` takes
    it — so ``ready + in-flight`` never exceeds ``size`` and the fill cursor
    stays within ``size`` of how many pops have happened.
    """

    def __init__(self, schedule: list[Task], backend: str | None, registry: SnapshotRegistry | None, size: int) -> None:
        self._schedule = schedule
        self._backend = backend
        self._registry = registry
        self._size = size

        self._cond = threading.Condition()
        self._ready: dict[str, list[Sandbox]] = {}
        self._pending: dict[str, int] = {}
        self._cursor = 0
        self._free_slots = size
        self._stop = False
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for i in range(self._size):
            t = threading.Thread(target=self._fill_loop, name=f"warm-queue-fill-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def pop(self, task: Task) -> Sandbox:
        """Return a ready sandbox for ``task``'s env, else create one synchronously.

        Blocks only while a fill for this exact env is in flight (so it waits
        for an almost-ready sandbox rather than spawning a duplicate); if none
        is ready or pending, it falls back to a direct ``get_sandbox`` at once.
        """
        key = env_key_for(task, self._backend)
        with self._cond:
            while not self._ready.get(key) and self._pending.get(key, 0) > 0 and not self._stop:
                self._cond.wait()
            ready = self._ready.get(key)
            if ready:
                sandbox = ready.pop()
                self._free_slots += 1
                self._cond.notify_all()
                return sandbox
        return get_sandbox(task, self._backend, self._registry)

    def shutdown(self) -> None:
        """Stop the fillers and close the prefetched-but-unconsumed tail.

        Setting ``_stop`` and snapshotting ``_ready`` happen under one lock, so
        every prefetched sandbox is closed exactly once: those already parked
        here, and any an in-flight filler is about to produce (it observes
        ``_stop`` and closes its own).
        """
        with self._cond:
            if self._stop:
                return
            self._stop = True
            self._cond.notify_all()
            leftover = [sandbox for sandboxes in self._ready.values() for sandbox in sandboxes]
            self._ready.clear()
        for t in self._threads:
            t.join()
        for sandbox in leftover:
            _close(sandbox)

    def _claim_next(self) -> tuple[Task, str] | None:
        """Claim the next schedule item and a slot, or ``None`` when stopped/exhausted."""
        with self._cond:
            while not self._stop:
                if self._cursor >= len(self._schedule):
                    return None
                if self._free_slots > 0:
                    task = self._schedule[self._cursor]
                    self._cursor += 1
                    self._free_slots -= 1
                    key = env_key_for(task, self._backend)
                    self._pending[key] = self._pending.get(key, 0) + 1
                    return task, key
                self._cond.wait()
            return None

    def _fill_loop(self) -> None:
        while True:
            claim = self._claim_next()
            if claim is None:
                return
            task, key = claim
            sandbox = None
            try:
                sandbox = get_sandbox(task, self._backend, self._registry)
            except Exception:
                logger.exception("warm queue: prefetch failed for task %s", getattr(task, "id", "?"))
            with self._cond:
                self._pending[key] -= 1
                parked = sandbox is not None and not self._stop
                if parked:
                    self._ready.setdefault(key, []).append(sandbox)
                else:
                    self._free_slots += 1
                self._cond.notify_all()
            if sandbox is not None and not parked:
                _close(sandbox)  # built, but a stop raced in first


__all__ = ["WarmQueue"]
