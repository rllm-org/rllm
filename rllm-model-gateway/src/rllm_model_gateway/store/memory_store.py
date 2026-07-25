"""In-memory trace store for testing and embedded usage."""

import array
import time
from collections import defaultdict
from typing import Any

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

    #: Cap on remembered tombstones. A deleted session is tombstoned
    #: *permanently* — session ids are unique uuids that are never legitimately
    #: reused, so any later write is a straggler that must never resurrect the
    #: session. Traces persist fire-and-forget, so a completion still in flight
    #: when the rollout ended can land *after* the trainer's ``delete_session``;
    #: without a tombstone ``store_trace`` re-creates a ``defaultdict`` entry that
    #: nothing ever deletes again, leaking the session for the process lifetime.
    #: The cap is only a memory backstop for long-lived multi-run gateways: it is
    #: far larger than the number of sessions deletable while any request is still
    #: in flight, so FIFO eviction never exposes a live straggler. 0 disables.
    _DEFAULT_MAX_TOMBSTONES = 1_000_000

    def __init__(self, max_tombstones: int = _DEFAULT_MAX_TOMBSTONES) -> None:
        # trace_id -> data dict
        self._traces: dict[str, dict[str, Any]] = {}
        # trace_id -> created_at
        self._timestamps: dict[str, float] = {}
        # session_id -> list[trace_id]  (insertion order)
        self._session_index: dict[str, list[str]] = defaultdict(list)
        # deleted session ids (insertion-ordered, FIFO-evicted at the cap);
        # blocks post-delete resurrection by a straggler trace
        self._tombstones: dict[str, None] = {}
        self._max_tombstones = max_tombstones

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        # A session id is deleted at most once and never reused, so any write to
        # a tombstoned session is a straggler that must not resurrect it.
        if session_id in self._tombstones:
            return
        now = time.time()
        self._traces[trace_id] = _pack(data)
        if trace_id not in self._timestamps:
            self._timestamps[trace_id] = now
        idx = self._session_index[session_id]
        if trace_id not in idx:
            idx.append(trace_id)

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        data = self._traces.get(trace_id)
        return _unpack(data) if data is not None else None

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
        return [_unpack(d) for d in results]

    async def delete_session(self, session_id: str) -> int:
        if self._max_tombstones != 0:
            # Tombstone permanently; evict oldest (FIFO) only to honour the cap.
            self._tombstones.pop(session_id, None)
            self._tombstones[session_id] = None
            while len(self._tombstones) > self._max_tombstones:
                self._tombstones.pop(next(iter(self._tombstones)))
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
