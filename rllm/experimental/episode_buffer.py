"""Episode buffer protocol and asyncio implementation for async training.

The buffer is a dumb pipe — no staleness filtering. Staleness is checked
at consumption time by the training loop. Backpressure is managed externally
by SyncCoordinator's rollout quota.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rllm.agents.agent import Episode


@dataclass
class BufferedEpisode:
    """A single episode stored in the buffer."""

    episode: Episode
    policy_version: int
    task: dict  # original task dict (for requeuing)
    task_id: str  # denormalized from episode.task_id


class EpisodeBufferProtocol(ABC):
    """Abstract base class for episode buffers.

    Different backends can provide different implementations:
    - AsyncioEpisodeBuffer: Single-threaded asyncio.Queue for Tinker
    - RayEpisodeBuffer (future): Ray actor for multi-process Verl
    """

    @abstractmethod
    async def put(self, item: BufferedEpisode) -> None:
        """Add an episode to the buffer."""

    @abstractmethod
    async def get(self) -> BufferedEpisode | None:
        """Get next episode. Returns None when generation is done and buffer is empty."""

    @abstractmethod
    def mark_generation_complete(self) -> None:
        """Signal that generation is finished."""

    @abstractmethod
    def qsize(self) -> int:
        """Current number of episodes in the buffer."""

    @abstractmethod
    def stats(self) -> dict:
        """Buffer statistics for metrics."""


class AsyncioEpisodeBuffer(EpisodeBufferProtocol):
    """Unbounded asyncio.Queue-based buffer for Tinker backend.

    No staleness filtering — that happens at consumption time.
    No max queue size — quota controls growth externally via SyncCoordinator.
    Tinker's compute happens on remote servers, so the Python process only
    orchestrates — no threading needed.
    """

    def __init__(self):
        self._queue: asyncio.Queue[BufferedEpisode | None] = asyncio.Queue()  # unbounded
        self._generation_complete = False
        self._total_produced = 0
        self._total_consumed = 0

    async def put(self, item: BufferedEpisode) -> None:
        await self._queue.put(item)
        self._total_produced += 1

    async def get(self) -> BufferedEpisode | None:
        while True:
            if self._generation_complete and self._queue.empty():
                return None
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._generation_complete and self._queue.empty():
                    return None
                continue
            if item is None:  # sentinel
                return None
            self._total_consumed += 1
            return item

    def mark_generation_complete(self) -> None:
        self._generation_complete = True

    def qsize(self) -> int:
        return self._queue.qsize()

    def stats(self) -> dict:
        return {
            "async/episode_buffer_size": self._queue.qsize(),
            "async/total_produced": self._total_produced,
            "async/total_consumed": self._total_consumed,
        }
