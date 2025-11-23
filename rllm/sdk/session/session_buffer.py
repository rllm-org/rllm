"""Session buffer for temporary trace storage during session lifecycle.

For persistent storage, see rllm.sdk.store (SqliteTraceStore).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Protocol, runtime_checkable

from rllm.sdk.protocol import Trace


@runtime_checkable
class SessionBufferProtocol(Protocol):
    """Protocol for session trace buffer backends."""

    def add_trace(self, session_uid_chain: list[str], session_name: str, trace: Trace) -> None:
        """Add trace to buffer under session hierarchy."""
        ...

    def get_traces(self, session_uid: str, session_name: str) -> list[Trace]:
        """Retrieve all traces for session (includes descendants)."""
        ...


class SessionBuffer:
    """Thread-safe in-memory trace buffer (default, single-process only)."""

    def __init__(self):
        self._traces: dict[str, list[Trace]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_trace(self, session_uid_chain: list[str], session_name: str, trace: Trace) -> None:
        """Add trace under all session UIDs in chain (enables parent queries)."""
        with self._lock:
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
            total = sum(len(t) for t in self._traces.values())
            return f"SessionBuffer(sessions={len(self._traces)}, traces={total})"
