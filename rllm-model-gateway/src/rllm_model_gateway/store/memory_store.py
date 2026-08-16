"""In-memory trace store for testing and embedded usage."""

import array
import time
from collections import defaultdict
from typing import Any

from rllm_model_gateway.models import TraceGraph, TraceRecord

_COMPACT_REQUIRED = frozenset({"messages", "response_message"})

# Token-id / logprob lists dominate trace memory. As Python list[int]/list[float]
# each element costs ~32-36 B (an 8 B pointer plus a boxed int/float object) since
# vocab ids exceed CPython's small-int intern cap (256). Packed into array.array
# they cost 4-8 B/element with no per-item object — ~9x smaller for ids, ~4x for
# logprobs — and collapse N objects into 1, cutting GC pressure. We pack on store
# and unpack on read so the external dict contract (plain lists) is unchanged.
_INT_ARRAY_FIELDS = ("prompt_token_ids", "completion_token_ids")
_FLOAT_ARRAY_FIELDS = ("logprobs",)


def _pack(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` with large id/logprob lists packed into ``array.array``.

    The dict is copied lazily (only if something is packed) so the caller's
    original lists can be released; any non-numeric/out-of-range content is left
    untouched, so this never changes what a reader sees after :func:`_unpack`.
    """
    out: dict[str, Any] | None = None
    for fields, typecode in ((_INT_ARRAY_FIELDS, "i"), (_FLOAT_ARRAY_FIELDS, "d")):
        for name in fields:
            value = data.get(name)
            if type(value) is not list or not value:
                continue
            try:
                packed = array.array(typecode, value)
            except (TypeError, ValueError, OverflowError):
                continue  # non-numeric / out-of-range: keep the original list
            if out is None:
                out = dict(data)
            out[name] = packed
    return out if out is not None else data


def _unpack(data: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`_pack` — restore packed arrays to plain lists."""
    out: dict[str, Any] | None = None
    for name in (*_INT_ARRAY_FIELDS, *_FLOAT_ARRAY_FIELDS):
        value = data.get(name)
        if isinstance(value, array.array):
            if out is None:
                out = dict(data)
            out[name] = value.tolist()
    return out if out is not None else data


class MemoryTraceStore:
    """Ephemeral in-memory store.  Useful for tests and short-lived processes."""

    def __init__(self, compact: bool = False) -> None:
        self._compact = compact
        self._traces: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}
        self._session_index: dict[str, list[str]] = defaultdict(list)
        self._session_seen: dict[str, set[str]] = defaultdict(set)
        self._graphs: dict[str, TraceGraph] = {}
        self._trace_session: dict[str, str] = {}

    @staticmethod
    def _record_from(trace_id: str, session_id: str, data: dict[str, Any]) -> TraceRecord:
        missing = _COMPACT_REQUIRED - data.keys()
        if missing:
            raise ValueError(f"compact trace missing required fields: {', '.join(sorted(missing))}")
        return TraceRecord.model_validate({**data, "trace_id": trace_id, "session_id": session_id})

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        now = time.time()
        if self._compact:
            record = self._record_from(trace_id, session_id, data)
            existing_session = self._trace_session.get(trace_id)
            if existing_session is not None and existing_session != session_id:
                raise ValueError(f"trace id {trace_id!r} belongs to another session")
            graph = self._graphs.get(session_id)
            if graph is None:
                graph = self._graphs[session_id] = TraceGraph(format="compact", version=1, deltas=[])
            if existing_session is not None:
                graph.replace_leaf(record)
                return
            graph.add(record)
            self._trace_session[trace_id] = session_id
            self._timestamps[trace_id] = now
            self._session_index[session_id].append(trace_id)
            return
        self._traces[trace_id] = _pack(data)
        if trace_id not in self._timestamps:
            self._timestamps[trace_id] = now
        if trace_id not in self._session_seen[session_id]:
            self._session_seen[session_id].add(trace_id)
            self._session_index[session_id].append(trace_id)

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        if self._compact:
            session_id = self._trace_session.get(trace_id)
            return None if session_id is None else self._graphs[session_id].resolve(trace_id).model_dump()
        data = self._traces.get(trace_id)
        return _unpack(data) if data is not None else None

    def _select(self, session_id: str, since: float | None, limit: int | None) -> list[str]:
        tids = [tid for tid in self._session_index.get(session_id, []) if since is None or self._timestamps.get(tid, 0.0) >= since]
        return tids[:limit]

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._compact:
            graph = self._graphs.get(session_id)
            return [] if graph is None else [graph.resolve(tid).model_dump() for tid in self._select(session_id, since, limit)]
        return [_unpack(self._traces[tid]) for tid in self._select(session_id, since, limit) if tid in self._traces]

    async def get_session_traces_compact(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return the compact session graph, optionally sliced."""
        if not self._compact:
            raise ValueError("compact traces require MemoryTraceStore(compact=True)")
        tids = self._select(session_id, since, limit)
        graph = self._graphs.get(session_id)
        if graph is None:
            graph = TraceGraph(format="compact", version=1, deltas=[])
        return graph.slice(tids).model_dump()

    async def count_session_traces(self, session_id: str) -> int:
        return len(self._session_index.get(session_id, []))

    async def delete_session(self, session_id: str) -> int:
        ids = self._session_index.pop(session_id, [])
        self._session_seen.pop(session_id, None)
        if self._compact:
            self._graphs.pop(session_id, None)
            for tid in ids:
                self._trace_session.pop(tid, None)
                self._timestamps.pop(tid, None)
            return len(ids)
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
