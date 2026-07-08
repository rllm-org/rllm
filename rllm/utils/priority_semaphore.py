"""A priority-aware asyncio semaphore.

Behaves like :class:`asyncio.Semaphore`, except that when a permit is released
and multiple coroutines are waiting, the one with the highest ``priority`` is
served first (FIFO within a priority). This lets eval rollouts preempt training
rollouts for slots on a *shared* concurrency pool: eval acquires at
``EVAL_PRIORITY`` and always wins contention, while training (``TRAIN_PRIORITY``)
naturally fills whatever capacity eval leaves free.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager

# Higher value == served first.
TRAIN_PRIORITY = 0
EVAL_PRIORITY = 1


class PrioritySemaphore:
    """An asyncio counting semaphore that grants waiters in priority order."""

    def __init__(self, value: int = 1):
        if value < 0:
            raise ValueError("PrioritySemaphore initial value must be >= 0")
        self._value = value
        # priority -> FIFO queue of waiter futures. Kept sparse (empty buckets
        # are dropped) so `waiting` and `_wake_up_next` stay cheap.
        self._waiters: dict[int, deque[asyncio.Future]] = {}

    @property
    def available(self) -> int:
        """Free permits not currently held."""
        return max(0, self._value)

    @property
    def waiting(self) -> int:
        """Number of coroutines queued waiting for a permit."""
        return sum(len(q) for q in self._waiters.values())

    def locked(self) -> bool:
        return self._value == 0

    async def acquire(self, priority: int = TRAIN_PRIORITY) -> bool:
        """Acquire a permit, blocking until one is free.

        When contending for a freed permit, higher ``priority`` waiters are
        served before lower ones; ties break FIFO.
        """
        if self._value > 0:
            self._value -= 1
            return True

        fut = asyncio.get_running_loop().create_future()
        q = self._waiters.setdefault(priority, deque())
        q.append(fut)
        try:
            try:
                await fut
            finally:
                # Woken (result set) or cancelled: drop ourselves from the queue.
                if fut in q:
                    q.remove(fut)
                if not q:
                    self._waiters.pop(priority, None)
        except asyncio.CancelledError:
            if not fut.cancelled():
                # release() granted us a permit (decrementing the counter on our
                # behalf) but we were cancelled before resuming. Return the permit
                # to the pool, then hand it to the next waiter instead of losing it.
                self._value += 1
                self._wake_up_next()
            raise
        return True

    def release(self) -> None:
        """Return a permit, waking the highest-priority waiter if any."""
        self._value += 1
        self._wake_up_next()

    def _wake_up_next(self) -> None:
        # Highest priority first; FIFO within a priority. On grant, decrement
        # the counter on the waiter's behalf so it holds the permit on resume.
        for priority in sorted(self._waiters, reverse=True):
            for fut in self._waiters[priority]:
                if not fut.done():
                    self._value -= 1
                    fut.set_result(True)
                    return

    @asynccontextmanager
    async def slot(self, priority: int = TRAIN_PRIORITY):
        """``async with sem.slot(priority): ...`` — acquire then release."""
        await self.acquire(priority)
        try:
            yield
        finally:
            self.release()
