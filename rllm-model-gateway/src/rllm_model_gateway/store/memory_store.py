"""In-memory trace store. Test fixture; not for production."""

from __future__ import annotations

import time
from typing import Any

from rllm_model_gateway.trace import TraceRecord


class MemoryTraceStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._traces: dict[str, TraceRecord] = {}
        self._extras: dict[str, tuple[str, bytes]] = {}

    # -- Sessions --------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        existing = self._sessions.get(session_id)
        if existing is None:
            self._sessions[session_id] = {
                "session_id": session_id,
                "created_at": time.time(),
                "metadata": metadata or {},
                "sampling_params": sampling_params or {},
            }
        else:
            if metadata:
                existing["metadata"] = {**existing.get("metadata", {}), **metadata}
            if sampling_params:
                existing["sampling_params"] = {**existing.get("sampling_params", {}), **sampling_params}

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        info = self._sessions.get(session_id)
        if info is None:
            return None
        info = dict(info)
        info["trace_count"] = sum(1 for t in self._traces.values() if t.session_id == session_id)
        return info

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for sid, info in self._sessions.items():
            if since is not None and info["created_at"] < since:
                continue
            entry = dict(info)
            entry["trace_count"] = sum(1 for t in self._traces.values() if t.session_id == sid)
            results.append(entry)
        results.sort(key=lambda r: r["created_at"], reverse=True)
        if limit is not None:
            results = results[:limit]
        return results

    async def delete_session(self, session_id: str) -> int:
        self._sessions.pop(session_id, None)
        to_drop = [tid for tid, t in self._traces.items() if t.session_id == session_id]
        for tid in to_drop:
            self._traces.pop(tid, None)
            self._extras.pop(tid, None)
        return len(to_drop)

    # -- Traces ----------------------------------------------------------

    async def store_trace(
        self,
        trace: TraceRecord,
        extras: tuple[str, bytes] | None = None,
    ) -> None:
        if trace.session_id not in self._sessions:
            raise KeyError(f"Session {trace.session_id!r} not found. Create it first via create_session().")
        self._traces[trace.trace_id] = trace
        if extras is not None:
            self._extras[trace.trace_id] = extras

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces.get(trace_id)

    async def get_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[TraceRecord]:
        results = [t for t in self._traces.values() if t.session_id == session_id]
        if since is not None:
            results = [t for t in results if t.timestamp >= since]
        results.sort(key=lambda t: t.timestamp)
        if limit is not None:
            results = results[:limit]
        return results

    async def get_trace_extras(self, trace_id: str) -> tuple[str, bytes] | None:
        return self._extras.get(trace_id)

    # -- Lifecycle -------------------------------------------------------

    async def flush(self) -> None: ...
    async def close(self) -> None: ...
