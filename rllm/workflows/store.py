"""Cross-episode key-value store for sharing state across workflow instances.

A :class:`Store` is shared across all workflow instances in a training run.
It persists across episodes and batches, enabling cross-episode information
sharing (e.g., reflection memory in ERL, strategy accumulation).

Basic usage::

    from rllm.workflows.store import InMemoryStore

    store = InMemoryStore()
    trainer = AgentTrainer(..., store=store)
    trainer.train()

Inside a workflow::

    async def run(self, task, uid, **kwargs):
        # read from the shared store
        prev = await self.store.get("improved_prompt")
        ...
        # write back
        await self.store.set("improved_prompt", new_prompt)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

_SENTINEL = object()


class Store(ABC):
    """Abstract base class for cross-episode key-value stores.

    Implementations must be safe for concurrent async access, since
    multiple workflow instances may read/write simultaneously from
    coroutines on the same event loop.
    """

    @abstractmethod
    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key. Returns *default* if not found."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        """Store a value under *key*. Overwrites any existing value."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key. Returns ``True`` if the key existed."""
        ...

    @abstractmethod
    async def keys(self) -> list[str]:
        """Return all keys currently in the store."""
        ...

    async def has(self, key: str) -> bool:
        """Check whether *key* exists in the store."""
        return await self.get(key, _SENTINEL) is not _SENTINEL

    async def clear(self) -> None:
        """Remove all entries from the store."""
        for k in list(await self.keys()):
            await self.delete(k)


class InMemoryStore(Store):
    """Thread-safe in-memory key-value store.

    Uses an :class:`asyncio.Lock` to serialise concurrent access from
    coroutines on the same event loop.  Suitable for single-process
    training (both Tinker and single-node Verl).
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock: asyncio.Lock | None = None

    @property
    def _async_lock(self) -> asyncio.Lock:
        # Lazy init: the Lock must be created inside the event loop that
        # uses it.  Since the Store may be instantiated *before*
        # asyncio.run(), we defer creation to first access.
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._async_lock:
            return self._data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        async with self._async_lock:
            self._data[key] = value

    async def delete(self, key: str) -> bool:
        async with self._async_lock:
            return self._data.pop(key, _SENTINEL) is not _SENTINEL

    async def keys(self) -> list[str]:
        async with self._async_lock:
            return list(self._data.keys())
