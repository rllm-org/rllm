"""In-memory trace store for testing and embedded usage."""

import array
import hashlib
import json
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


# Sentinel key marking an interned ``messages`` field: [session_id, leaf_node_id, length].
# Present only inside the store; readers always see the expanded list.
_INTERNED_KEY = "__interned_messages__"


def _message_fp(message: dict[str, Any]) -> str:
    """Stable content fingerprint of one message (canonical JSON → sha256)."""
    payload = json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class MemoryTraceStore:
    """Ephemeral in-memory store.  Useful for tests and short-lived processes.

    ``compact=True`` (the ``memory-compact`` store worker) opts into message
    interning: agentic clients resend the whole conversation on every call, so
    trace N of a session carries the same N-1 leading messages as trace N-1 —
    storing each trace's ``messages`` verbatim makes a session's memory
    quadratic in turns (measured 30–400 MB per task on DeepSWE-length
    rollouts). Like ``_pack`` does for token-id lists, compact mode interns on
    write: each unique (parent, content) pair is kept once in a per-session
    node table and the trace stores only its leaf reference; reads expand back
    to the full list, so the external contract is unchanged. Node identity
    hashes the *parent chain* as well as content (a Merkle chain) so identical
    content at different positions — e.g. repeated empty completions from
    throttled calls — keeps distinct nodes and reconstruction is exact.

    The default mode stores traces verbatim, exactly as before.
    """

    def __init__(self, compact: bool = False) -> None:
        self._compact = compact
        # trace_id -> data dict
        self._traces: dict[str, dict[str, Any]] = {}
        # trace_id -> created_at
        self._timestamps: dict[str, float] = {}
        # session_id -> list[trace_id]  (insertion order)
        self._session_index: dict[str, list[str]] = defaultdict(list)
        # session_id -> {node_id: (parent_node_id | None, message)}
        self._session_nodes: dict[str, dict[str, tuple[str | None, dict[str, Any]]]] = defaultdict(dict)
        # session_id -> {(parent_node_id | None, content_fp): node_id} — O(1)
        # re-interning of prefixes without walking or re-storing them.
        self._session_chain: dict[str, dict[tuple[str | None, str], str]] = defaultdict(dict)

    def _intern_messages(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Return ``data`` with ``messages`` replaced by a leaf reference."""
        messages = data.get("messages")
        if type(messages) is not list or not messages or not all(isinstance(m, dict) for m in messages):
            return data  # empty/unknown shape: store verbatim
        nodes = self._session_nodes[session_id]
        chain = self._session_chain[session_id]
        parent: str | None = None
        for message in messages:
            key = (parent, _message_fp(message))
            node_id = chain.get(key)
            if node_id is None:
                # Merkle id: parent chain + content, so position matters.
                node_id = hashlib.sha256(f"{parent}:{key[1]}".encode()).hexdigest()
                nodes[node_id] = (parent, message)
                chain[key] = node_id
            parent = node_id
        out = dict(data)
        out["messages"] = {_INTERNED_KEY: [session_id, parent, len(messages)]}
        return out

    def _expand_messages(self, data: dict[str, Any]) -> dict[str, Any]:
        """Inverse of :meth:`_intern_messages` — rebuild the full message list."""
        marker = data.get("messages")
        if type(marker) is not dict or _INTERNED_KEY not in marker:
            return data
        session_id, leaf, length = marker[_INTERNED_KEY]
        nodes = self._session_nodes.get(session_id, {})
        messages: list[dict[str, Any]] = []
        node_id = leaf
        while node_id is not None:
            parent, message = nodes[node_id]
            messages.append(message)
            node_id = parent
        messages.reverse()
        assert len(messages) == length, f"interned chain length {len(messages)} != recorded {length}"
        out = dict(data)
        out["messages"] = messages
        return out

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        now = time.time()
        if self._compact:
            data = self._intern_messages(session_id, data)
        self._traces[trace_id] = _pack(data)
        if trace_id not in self._timestamps:
            self._timestamps[trace_id] = now
        idx = self._session_index[session_id]
        if trace_id not in idx:
            idx.append(trace_id)

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        data = self._traces.get(trace_id)
        return self._expand_messages(_unpack(data)) if data is not None else None

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
        return [self._expand_messages(_unpack(d)) for d in results]

    async def delete_session(self, session_id: str) -> int:
        ids = self._session_index.pop(session_id, [])
        self._session_nodes.pop(session_id, None)
        self._session_chain.pop(session_id, None)
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
