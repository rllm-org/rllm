"""Shared scoring accumulator for batching teacher-scoring calls across workflows.

When multiple :class:`GsdWorkflow` instances run concurrently, each
independently calls :func:`score_teacher_for_response` for its own
trajectories.  The ``ScoringAccumulator`` collects these coroutines and
fires them in larger batches, improving Tinker server utilisation.

Usage::

    from rllm.experimental.gsd.scoring_accumulator import ScoringAccumulator

    accumulator = ScoringAccumulator(batch_interval=0.05, batch_threshold=64)
    # pass via workflow_args so every GsdWorkflow instance shares it
    trainer = AgentTrainer(
        workflow_args={"scoring_accumulator": accumulator, ...},
        ...
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class ScoringAccumulator:
    """Batches scoring coroutines across concurrent workflows.

    Coroutines submitted via :meth:`submit` are collected and fired together
    in one ``asyncio.gather`` call.  A batch is dispatched when **either**:

    * ``batch_threshold`` requests have accumulated (low-latency path), **or**
    * ``batch_interval`` seconds have elapsed since the first request in the
      current window (prevents indefinite waiting when few workflows score).

    Parameters
    ----------
    batch_interval:
        Maximum seconds to wait before flushing a partial batch.
    batch_threshold:
        Flush immediately when this many coroutines are pending.
    """

    def __init__(
        self,
        batch_interval: float = 0.05,
        batch_threshold: int = 64,
    ) -> None:
        self._batch_interval = batch_interval
        self._batch_threshold = batch_threshold

        self._pending: list[tuple[asyncio.Future[Any], Coroutine[Any, Any, Any]]] = []
        self._lock = asyncio.Lock()
        self._flush_scheduled = False

        # Metrics (non-critical, no lock needed — single event loop)
        self.total_submitted: int = 0
        self.total_batches: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Submit a scoring coroutine and wait for its result.

        The caller blocks until the batch containing this coroutine fires
        and the result is available.  Multiple concurrent callers will have
        their coroutines batched together.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        fire_now = False
        async with self._lock:
            self._pending.append((future, coro))
            self.total_submitted += 1

            if len(self._pending) >= self._batch_threshold:
                fire_now = True
            elif not self._flush_scheduled:
                self._flush_scheduled = True
                loop.create_task(self._delayed_flush())

        if fire_now:
            await self._flush()

        return await future

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _delayed_flush(self) -> None:
        """Timer-triggered flush after ``batch_interval`` seconds."""
        await asyncio.sleep(self._batch_interval)
        await self._flush()

    async def _flush(self) -> None:
        """Collect all pending coroutines and fire them in one gather."""
        async with self._lock:
            if not self._pending:
                self._flush_scheduled = False
                return
            batch = self._pending.copy()
            self._pending.clear()
            self._flush_scheduled = False

        batch_size = len(batch)
        self.total_batches += 1
        t0 = time.monotonic()

        coros = [coro for _, coro in batch]
        results = await asyncio.gather(*coros, return_exceptions=True)

        elapsed = time.monotonic() - t0
        logger.debug(f"[ScoringAccumulator] flushed batch #{self.total_batches}: {batch_size} coros in {elapsed:.2f}s")

        for (future, _), result in zip(batch, results, strict=False):
            if future.done():
                # Should not happen, but guard against double-resolve
                continue
            if isinstance(result, BaseException):
                future.set_exception(result)
            else:
                future.set_result(result)
