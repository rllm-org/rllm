"""In-memory trace store for testing and embedded usage."""

import time
from collections import defaultdict
from typing import Any


class MemoryTraceStore:
    """Ephemeral in-memory store. Useful for tests and short-lived processes.

    The session index is keyed by ``(session_id, run_id)`` so the same
    session_id can coexist across runs without collision (e.g. eval-0
    from two concurrent eval invocations).
    """

    def __init__(self) -> None:
        # trace_id -> data dict
        self._traces: dict[str, dict[str, Any]] = {}
        # trace_id -> created_at
        self._timestamps: dict[str, float] = {}
        # (session_id, run_id) -> list[trace_id]  (insertion order)
        self._session_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        # run_id -> {metadata, started_at, ended_at}
        self._runs: dict[str, dict[str, Any]] = {}

    async def store_trace(
        self,
        trace_id: str,
        session_id: str,
        data: dict[str, Any],
        run_id: str = "",
    ) -> None:
        now = time.time()
        self._traces[trace_id] = data
        if trace_id not in self._timestamps:
            self._timestamps[trace_id] = now
        idx = self._session_index[(session_id, run_id or "")]
        if trace_id not in idx:
            idx.append(trace_id)

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self._traces.get(trace_id)

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for (sid, rid), tids in self._session_index.items():
            if sid != session_id:
                continue
            if run_id is not None and rid != run_id:
                continue
            for tid in tids:
                ts = self._timestamps.get(tid, 0.0)
                if since is not None and ts < since:
                    continue
                data = self._traces.get(tid)
                if data is not None:
                    results.append(data)
        if limit is not None:
            results = results[:limit]
        return results

    async def delete_session(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
    ) -> int:
        keys_to_delete: list[tuple[str, str]] = []
        for key in list(self._session_index.keys()):
            sid, rid = key
            if sid != session_id:
                continue
            if run_id is not None and rid != run_id:
                continue
            keys_to_delete.append(key)

        deleted = 0
        for key in keys_to_delete:
            ids = self._session_index.pop(key, [])
            referenced: set[str] = set()
            for tids in self._session_index.values():
                referenced.update(tids)
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
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for (sid, rid), tids in self._session_index.items():
            if run_id is not None and rid != run_id:
                continue
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
                    "run_id": rid,
                    "trace_count": len(tids),
                    "first_trace_at": first_at,
                    "last_trace_at": max(timestamps),
                }
            )
        results.sort(key=lambda r: r["first_trace_at"], reverse=True)
        if limit is not None:
            results = results[:limit]
        return results

    async def register_run(
        self,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        existing = self._runs.get(run_id)
        if existing is None:
            self._runs[run_id] = {
                "metadata": dict(metadata or {}),
                "started_at": time.time(),
                "ended_at": None,
            }
        else:
            # Idempotent: refresh metadata, clear ended_at, keep started_at.
            existing["metadata"] = dict(metadata or {})
            existing["ended_at"] = None

    async def end_run(self, run_id: str) -> None:
        info = self._runs.get(run_id)
        if info is not None and info.get("ended_at") is None:
            info["ended_at"] = time.time()

    async def list_runs(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for rid, info in self._runs.items():
            started = info["started_at"]
            if since is not None and started < since:
                continue
            results.append(
                {
                    "run_id": rid,
                    "started_at": started,
                    "ended_at": info["ended_at"],
                    "metadata": dict(info["metadata"]),
                }
            )
        results.sort(key=lambda r: r["started_at"], reverse=True)
        if limit is not None:
            results = results[:limit]
        return results

    async def flush(self) -> None:
        """No-op for in-memory store."""

    async def close(self) -> None:
        """No-op for in-memory store."""
