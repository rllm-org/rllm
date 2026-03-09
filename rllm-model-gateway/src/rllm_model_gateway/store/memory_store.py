"""In-memory trace store for testing and embedded usage."""

import time
from collections import defaultdict
from typing import Any


class MemoryTraceStore:
    """Ephemeral in-memory store.  Useful for tests and short-lived processes."""

    def __init__(self) -> None:
        # trace_id -> data dict
        self._traces: dict[str, dict[str, Any]] = {}
        # trace_id -> created_at
        self._timestamps: dict[str, float] = {}
        # session_id -> list[trace_id]  (insertion order)
        self._session_index: dict[str, list[str]] = defaultdict(list)

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        now = time.time()
        self._traces[trace_id] = data
        if trace_id not in self._timestamps:
            self._timestamps[trace_id] = now
        idx = self._session_index[session_id]
        if trace_id not in idx:
            idx.append(trace_id)

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self._traces.get(trace_id)

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        ids = self._session_index.get(session_id, [])
        results: list[dict[str, Any]] = []
        for tid in ids:
            ts = self._timestamps.get(tid, 0.0)
            if since is not None and ts < since:
                continue
            data = self._traces.get(tid)
            if data is not None:
                results.append(data)
        if limit is not None:
            results = results[:limit]
        return results

    async def delete_session(self, session_id: str) -> int:
        ids = self._session_index.pop(session_id, [])
        # Collect trace_ids referenced by other sessions
        referenced: set[str] = set()
        for sid, tids in self._session_index.items():
            referenced.update(tids)
        deleted = 0
        for tid in ids:
            if tid not in referenced:
                self._traces.pop(tid, None)
                self._timestamps.pop(tid, None)
                deleted += 1
        return deleted

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for sid, tids in self._session_index.items():
            if not tids:
                continue
            timestamps = [self._timestamps[t] for t in tids if t in self._timestamps]
            if not timestamps:
                continue
            first_at = min(timestamps)
            if since is not None and first_at < since:
                continue
            results.append(
                {
                    "session_id": sid,
                    "trace_count": len(tids),
                    "first_trace_at": first_at,
                    "last_trace_at": max(timestamps),
                }
            )
        results.sort(key=lambda r: r["first_trace_at"], reverse=True)
        if limit is not None:
            results = results[:limit]
        return results

    async def flush(self) -> None:
        """No-op for in-memory store."""

    async def close(self) -> None:
        """No-op for in-memory store."""
