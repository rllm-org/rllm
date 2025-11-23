"""Storage protocols and implementations for session trace storage."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Protocol, runtime_checkable

from asgiref.sync import async_to_sync

from rllm.sdk.protocol import Trace

logger = logging.getLogger(__name__)


@runtime_checkable
class SessionStorage(Protocol):
    """Protocol for session trace storage backends.

    Implementations: InMemoryStorage (single-process), SqliteSessionStorage (multi-process).
    Uses session_uid_chain for hierarchy support - parent sessions see all descendant traces.
    """

    def add_trace(self, session_uid_chain: list[str], session_name: str, trace: Trace) -> None:
        """Add trace to storage under session hierarchy."""
        ...

    def get_traces(self, session_uid: str, session_name: str) -> list[Trace]:
        """Retrieve all traces for session (includes descendants)."""
        ...


class InMemoryStorage:
    """Thread-safe in-memory trace storage (default, single-process only).

    Fast ephemeral storage. For multi-process use SqliteSessionStorage.
    """

    def __init__(self):
        """Initialize thread-safe in-memory storage."""
        self._traces: dict[str, list[Trace]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_trace(self, session_uid_chain: list[str], session_name: str, trace: Trace) -> None:
        """
        Add trace to in-memory storage (thread-safe).

        Stores the trace under ALL session UIDs in the chain, enabling
        parent sessions to query all descendant traces.

        Args:
            session_uid_chain: List of session UIDs from root to current
            session_name: User-visible session name (for logging/debugging)
            trace: Trace object to store
        """
        with self._lock:
            # Store trace under all session UIDs in the chain
            # This enables tree queries: parent sees all descendant traces
            for uid in session_uid_chain:
                self._traces[uid].append(trace)

    def get_traces(self, session_uid: str, session_name: str) -> list[Trace]:
        """Get all traces for session UID (thread-safe)."""
        with self._lock:
            return self._traces[session_uid].copy()

    def clear(self, session_uid: str, session_name: str) -> None:
        """Clear all traces for session UID (thread-safe)."""
        with self._lock:
            if session_uid in self._traces:
                self._traces[session_uid].clear()

    def __repr__(self):
        with self._lock:
            total_traces = sum(len(traces) for traces in self._traces.values())
            return f"InMemoryStorage(sessions={len(self._traces)}, total_traces={total_traces})"


class SqliteSessionStorage:
    """SQLite-backed durable storage for multi-process trace sharing.

    Supports both synchronous and asynchronous APIs:
    - Sync: `add_trace()`, `get_traces()` - block until complete
    - Async: `async_add_trace()`, `async_get_traces()` - await directly

    Sync methods use asgiref.sync.async_to_sync for safe async-to-sync conversion.
    For optimal performance in async contexts, use async methods directly.
    """

    def __init__(self, db_path: str | None = None):
        """Initialize SQLite storage (defaults to ~/.rllm/traces.db)."""
        from rllm.sdk.store import SqliteTraceStore

        self.store = SqliteTraceStore(db_path=db_path)

    def add_trace(self, session_uid_chain: list[str], session_name: str, trace: Trace) -> None:
        """Add trace to SQLite (synchronous, blocks until write completes).

        This method ensures the trace is fully written to the database before returning,
        providing the same synchronous semantics as InMemoryStorage.

        For async contexts, prefer using `async_add_trace()` to avoid conversion overhead.

        Note: Uses asgiref.sync.async_to_sync for safe async-to-sync conversion.
        """
        async_to_sync(self.async_add_trace)(session_uid_chain, trace)

    async def async_add_trace(self, session_uid_chain: list[str], trace: Trace) -> None:
        """Add trace to SQLite (asynchronous, awaitable).

        This is the native async method that directly awaits the store operation.
        Use this in async contexts for better performance than the sync `add_trace()`.

        Args:
            session_uid_chain: List of session UIDs from root to current
            trace: Trace object to store

        Example:
            >>> storage = SqliteSessionStorage(db_path="./traces.db")
            >>> await storage.async_add_trace(session_uid_chain, trace)
        """
        try:
            await self.store.store(
                trace_id=trace.trace_id,
                data=trace.model_dump(),
                namespace="default",
                context_type="llm_trace",
                metadata={"session_name": trace.session_name},
                session_uids=session_uid_chain,  # Pass full chain to junction table!
            )
        except Exception as e:
            logger.exception(f"Failed to store trace {trace.trace_id}: {e}")

    def get_traces(self, session_uid: str, session_name: str) -> list[Trace]:
        """Retrieve all traces for a session from SQLite (synchronous).

        Uses session_uid to query the junction table, returning all traces
        stored under this UID (including descendant sessions in the tree).

        For async contexts, prefer using `async_get_traces()` to avoid conversion overhead.

        Args:
            session_uid: Unique session context UID (used as storage key)
            session_name: User-visible session name (for logging/debugging)

        Returns:
            List of Trace objects for this session and all descendants

        Note: Uses asgiref.sync.async_to_sync for safe async-to-sync conversion.
        """
        return async_to_sync(self.async_get_traces)(session_uid)

    async def async_get_traces(self, session_uid: str) -> list[Trace]:
        """Retrieve all traces for a session from SQLite (asynchronous, awaitable).

        This is the native async method that directly awaits the store operation.
        Use this in async contexts for better performance than the sync `get_traces()`.

        Args:
            session_uid: Session context UID used as storage key

        Returns:
            List of Trace objects for this session and all descendants

        Example:
            >>> storage = SqliteSessionStorage(db_path="./traces.db")
            >>> traces = await storage.async_get_traces(session_uid)
        """
        try:
            trace_contexts = await self.store.get_by_session_uid(session_uid)

            # Convert TraceContext objects to Trace protocol objects
            traces = []
            for tc in trace_contexts:
                # tc.data is already a dict with all the trace fields
                trace = Trace(**tc.data)
                traces.append(trace)

            return traces
        except Exception as e:
            logger.exception(f"Failed to retrieve traces for session UID {session_uid}: {e}")
            return []

    def __repr__(self):
        return f"SqliteSessionStorage(db_path={self.store.db_path!r})"
