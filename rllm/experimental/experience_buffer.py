"""Experience buffer protocol and asyncio implementation for async training."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rllm.agents.agent import Episode, TrajectoryGroup


@dataclass
class BufferedExperience:
    """A unit of experience stored in the buffer."""

    trajectory_groups: list[TrajectoryGroup]
    episodes: list[Episode]
    policy_version: int
    batch_source: Any  # Original batch (for requeuing)


class ExperienceBufferProtocol(ABC):
    """Abstract base class for experience buffers.

    Different backends can provide different implementations:
    - AsyncioExperienceBuffer: Single-threaded asyncio.Queue for Tinker
    - RayExperienceBuffer (future): Ray actor for multi-process Verl
    """

    @abstractmethod
    async def put(self, experience: BufferedExperience) -> None:
        """Add experience. Blocks if buffer full (backpressure)."""

    @abstractmethod
    async def get(self, current_policy_version: int) -> BufferedExperience | None:
        """Get next non-stale experience. Returns None when done."""

    @abstractmethod
    def mark_generation_complete(self) -> None:
        """Signal that generation is finished."""

    @abstractmethod
    async def get_requeue_batch(self) -> Any | None:
        """Get a stale batch to regenerate, or None."""

    @abstractmethod
    def stats(self) -> dict:
        """Buffer statistics for metrics."""


class AsyncioExperienceBuffer(ExperienceBufferProtocol):
    """Single-threaded asyncio-based buffer for Tinker backend.

    Uses asyncio.Queue for backpressure. Tinker's compute happens on remote
    servers, so the Python process only orchestrates — no threading needed.
    """

    def __init__(self, max_size: int, max_staleness: int, requeue_stale: bool):
        self._queue: asyncio.Queue[BufferedExperience | None] = asyncio.Queue(maxsize=max_size)
        self._requeue_queue: asyncio.Queue = asyncio.Queue()
        self._max_staleness = max_staleness
        self._requeue_stale = requeue_stale
        self._generation_complete = False
        # Stats
        self._total_produced = 0
        self._total_consumed = 0
        self._total_discarded = 0

    async def put(self, experience: BufferedExperience) -> None:
        await self._queue.put(experience)
        self._total_produced += 1

    async def get(self, current_policy_version: int) -> BufferedExperience | None:
        while True:
            if self._generation_complete and self._queue.empty():
                return None
            try:
                experience = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._generation_complete and self._queue.empty():
                    return None
                continue
            if experience is None:  # sentinel
                return None
            # Staleness check
            version_gap = current_policy_version - experience.policy_version
            if version_gap > self._max_staleness:
                self._total_discarded += 1
                if self._requeue_stale:
                    await self._requeue_queue.put(experience.batch_source)
                continue
            self._total_consumed += 1
            return experience

    def mark_generation_complete(self) -> None:
        self._generation_complete = True

    async def get_requeue_batch(self) -> Any | None:
        try:
            return self._requeue_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def stats(self) -> dict:
        return {
            "async/buffer_size": self._queue.qsize(),
            "async/total_produced": self._total_produced,
            "async/total_consumed": self._total_consumed,
            "async/total_discarded": self._total_discarded,
        }
