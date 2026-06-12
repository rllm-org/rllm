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

Two guarantees the queue maintains (beyond the original prefetch contract):

- **pop never hands out a dead sandbox.** Remote sandboxes can die while
  parked (provider idle auto-stop, lifetime timeout); pop asks the backend
  via ``sandbox.is_alive()`` before handing one over, and replaces a dead
  one transparently — the consumer just waits for the replacement. The
  freed slot lets the fillers prefetch ahead for the rest of the batch in
  the meantime.
- **misses never disturb the schedule.** A pop that self-serves inline
  (because its entry was never prefetched) leaves a *credit* so the filler
  skips the matching schedule entry instead of building a sandbox nobody
  will pop; a failed prefetch is retried once and, if it still fails, is
  remembered so the eventual pop miss doesn't mistakenly credit-skip a
  later entry of the same env.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rllm.sandbox.snapshot import env_key_for, get_sandbox

if TYPE_CHECKING:
    from rllm.sandbox.protocol import Sandbox
    from rllm.sandbox.snapshot import SnapshotRegistry
    from rllm.types import Task

logger = logging.getLogger(__name__)

# Backoff before the single prefetch retry; absorbs transient provider
# errors (create timeouts, rate limits) without turning fillers into
# retry loops.
_PREFETCH_RETRY_BACKOFF_S = 15.0


def _close(sandbox: Sandbox) -> None:
    try:
        sandbox.close()
    except Exception:
        logger.exception("warm queue: sandbox close failed")


@dataclass
class _WarmEntry:
    """A parked sandbox plus when it was parked (for diagnostics only)."""

    sandbox: Sandbox
    created_at: float


def _is_alive(sandbox: Sandbox) -> bool:
    """Ask the backend whether the sandbox is still usable.

    A backend without ``is_alive`` (e.g. a third-party Sandbox
    implementation predating the protocol method) is assumed alive —
    exactly the pre-check behavior. An ``is_alive`` that raises counts
    as dead: the implementations are documented not to raise, so an
    escape here means something is genuinely wrong with the sandbox.
    """
    check = getattr(sandbox, "is_alive", None)
    if not callable(check):
        return True
    try:
        return bool(check())
    except Exception:
        logger.warning("warm queue: is_alive check raised — treating sandbox as dead", exc_info=True)
        return False


class WarmQueue:
    """Prefetch buffer of ready sandboxes for a run's ordered schedule.

    All shared state lives under one :class:`threading.Condition`. A *slot* is
    held from the moment a filler claims a schedule item, through its
    ``get_sandbox`` call, while the sandbox sits ready, until :meth:`pop` takes
    it — so ``ready + in-flight`` never exceeds ``size`` and the fill cursor
    stays within ``size`` of how many pops have happened.
    """

    def __init__(self, schedule: list[Task], backend: str | None, registry: SnapshotRegistry | None, size: int, install_script: str = "") -> None:
        self._schedule = schedule
        self._backend = backend
        self._registry = registry
        self._size = size
        self._install_script = install_script

        self._cond = threading.Condition()
        self._ready: dict[str, list[_WarmEntry]] = {}
        self._pending: dict[str, int] = {}
        # Pops that self-served inline: the filler skips one matching
        # schedule entry per credit instead of building an orphan.
        self._credits: dict[str, int] = {}
        # Prefetches that failed for good: the matching pop's miss is
        # already balanced (the entry was consumed without a product), so
        # it must NOT leave a credit.
        self._failed: dict[str, int] = {}
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
        """Return a live sandbox for ``task``'s env, else create one synchronously.

        Blocks only while a fill for this exact env is in flight (so it waits
        for an almost-ready sandbox rather than spawning a duplicate); if none
        is ready or pending, it falls back to a direct ``get_sandbox`` at once.

        Every candidate is liveness-checked via the backend API before being
        handed over; a dead one is closed and replaced (next ready sandbox,
        the in-flight fill, or an inline create) within this call, so the
        consumer always receives a sandbox that was just confirmed alive.
        """
        key = env_key_for(task, self._backend, self._install_script)
        replacing = False
        while True:
            entry = None
            with self._cond:
                while not self._ready.get(key) and self._pending.get(key, 0) > 0 and not self._stop:
                    self._cond.wait()
                ready = self._ready.get(key)
                if ready:
                    entry = ready.pop()
                    self._free_slots += 1
                    self._cond.notify_all()
                elif not self._stop and not replacing:
                    # Pure miss: this entry was never prefetched. If a failed
                    # prefetch explains it, the schedule already moved past the
                    # entry — just consume the marker. Otherwise the entry is
                    # still ahead of the cursor: credit it so the filler skips
                    # it instead of building a sandbox nobody will pop.
                    if self._failed.get(key, 0) > 0:
                        self._failed[key] -= 1
                    else:
                        self._credits[key] = self._credits.get(key, 0) + 1
                        self._cond.notify_all()
            if entry is None:
                return get_sandbox(task, self._backend, self._registry, self._install_script)
            if _is_alive(entry.sandbox):  # backend API call, outside the lock
                return entry.sandbox
            replacing = True
            logger.warning("warm queue: sandbox for %s died while parked (%.0fs) — replacing it", key, time.monotonic() - entry.created_at)
            _close(entry.sandbox)
            # Loop: another ready sandbox or an in-flight fill may serve this
            # key; otherwise the inline fallback creates the replacement. The
            # slot this entry held is already free, so fillers prefetch ahead
            # while this consumer waits.

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
            leftover = [entry for entries in self._ready.values() for entry in entries]
            self._ready.clear()
        for t in self._threads:
            t.join()
        for entry in leftover:
            _close(entry.sandbox)

    def _claim_next(self) -> tuple[Task, str] | None:
        """Claim the next schedule item and a slot, or ``None`` when stopped/exhausted.

        Credited entries (their pop already self-served inline) are consumed
        without building — the cursor advances, no slot is taken.
        """
        with self._cond:
            while not self._stop:
                if self._cursor >= len(self._schedule):
                    return None
                task = self._schedule[self._cursor]
                key = env_key_for(task, self._backend, self._install_script)
                if self._credits.get(key, 0) > 0:
                    self._credits[key] -= 1
                    self._cursor += 1
                    logger.info("warm queue: skipping schedule entry for %s — its consumer already created inline", key)
                    continue
                if self._free_slots > 0:
                    self._cursor += 1
                    self._free_slots -= 1
                    self._pending[key] = self._pending.get(key, 0) + 1
                    return task, key
                self._cond.wait()
            return None

    def _wait_unless_stopped(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds, waking early on stop; return the stop flag."""
        with self._cond:
            if not self._stop:
                self._cond.wait(timeout=timeout)
            return self._stop

    def _fill_loop(self) -> None:
        while True:
            claim = self._claim_next()
            if claim is None:
                return
            task, key = claim
            sandbox = None
            for attempt in (1, 2):
                try:
                    sandbox = get_sandbox(task, self._backend, self._registry, self._install_script)
                    break
                except Exception:
                    logger.exception("warm queue: prefetch failed for task %s (attempt %d/2)", getattr(task, "id", "?"), attempt)
                    if attempt == 1 and self._wait_unless_stopped(_PREFETCH_RETRY_BACKOFF_S):
                        break
            with self._cond:
                self._pending[key] -= 1
                parked = sandbox is not None and not self._stop
                if parked:
                    self._ready.setdefault(key, []).append(_WarmEntry(sandbox, time.monotonic()))
                else:
                    self._free_slots += 1
                    if sandbox is None:
                        # Gave up on this entry: remember it so the matching
                        # pop miss doesn't credit-skip a later entry.
                        self._failed[key] = self._failed.get(key, 0) + 1
                self._cond.notify_all()
            if sandbox is not None and not parked:
                _close(sandbox)  # built, but a stop raced in first


__all__ = ["WarmQueue"]
