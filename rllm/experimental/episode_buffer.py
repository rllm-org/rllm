"""Episode buffer protocol and asyncio implementation for async training.

The buffer is a dumb pipe — no staleness filtering. Staleness is controlled
at dispatch time by SyncCoordinator's throttle quota.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rllm.agents.agent import Episode


@dataclass
class BufferedEpisodeGroup:
    """All n episodes for one prompt, collected before buffering."""

    episodes: list[Episode]
    weight_version: int  # earliest weight_version across all steps in all episodes
    task_id: str


class EpisodeGroupAccumulator:
    """Per-task collector that groups episodes by task_id before pushing to buffer.

    Lives in the generation loop, NOT inside the buffer (buffer stays a dumb pipe).
    Optionally filters out groups with no gradient signal (all correct or all incorrect).
    """

    def __init__(
        self,
        group_size: int,
        buffer: EpisodeBufferProtocol,
        filter_uniform_groups: bool = False,
        on_group_filtered: callable | None = None,
    ):
        self._group_size = group_size
        self._buffer = buffer
        self._filter_uniform_groups = filter_uniform_groups
        self._on_group_filtered = on_group_filtered
        self._pending: dict[str, list[Episode]] = {}
        self.total_filtered: int = 0

    async def add_episode(self, task_id: str, episode: Episode) -> bool:
        """Add episode. Returns True if group completed (pushed or filtered)."""
        self._pending.setdefault(task_id, []).append(episode)
        if len(self._pending[task_id]) == self._group_size:
            episodes = self._pending.pop(task_id)

            if self._filter_uniform_groups and len({ep.is_correct for ep in episodes}) == 1:
                self.total_filtered += 1
                if self._on_group_filtered:
                    self._on_group_filtered()
                return True

            earliest = self._compute_earliest_version(episodes)
            await self._buffer.put(BufferedEpisodeGroup(episodes=episodes, weight_version=earliest, task_id=task_id))
            return True
        return False

    @staticmethod
    def _compute_earliest_version(episodes: list[Episode]) -> int:
        min_v = float("inf")
        for ep in episodes:
            for traj in ep.trajectories:
                for step in traj.steps:
                    if step.weight_version is not None:
                        min_v = min(min_v, step.weight_version)
        return int(min_v) if min_v != float("inf") else 0


class EpisodeBufferProtocol(ABC):
    """Abstract base class for episode buffers.

    Different backends can provide different implementations:
    - AsyncioEpisodeBuffer: Single-threaded asyncio.Queue for Tinker
    - RayEpisodeBuffer (future): Ray actor for multi-process Verl
    """

    @abstractmethod
    async def put(self, item: BufferedEpisodeGroup) -> None:
        """Add an episode group to the buffer."""

    @abstractmethod
    async def get(self) -> BufferedEpisodeGroup | None:
        """Get next episode group. Returns None when generation is done and buffer is empty."""

    @abstractmethod
    def mark_generation_complete(self) -> None:
        """Signal that generation is finished."""

    @abstractmethod
    def qsize(self) -> int:
        """Current number of episode groups in the buffer."""

    @abstractmethod
    def stats(self) -> dict:
        """Buffer statistics for metrics."""


class AsyncioEpisodeBuffer(EpisodeBufferProtocol):
    """Unbounded asyncio.Queue-based buffer for Tinker backend.

    No staleness filtering — throttle controls growth externally via SyncCoordinator.
    Tinker's compute happens on remote servers, so the Python process only
    orchestrates — no threading needed.
    """

    def __init__(self):
        self._queue: asyncio.Queue[BufferedEpisodeGroup | None] = asyncio.Queue()  # unbounded
        self._generation_complete = False
        self._total_produced = 0
        self._total_consumed = 0

    async def put(self, item: BufferedEpisodeGroup) -> None:
        await self._queue.put(item)
        self._total_produced += 1

    async def get(self) -> BufferedEpisodeGroup | None:
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
