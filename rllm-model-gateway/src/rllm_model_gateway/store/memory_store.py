"""In-memory trace store for testing and embedded usage."""

import array
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from typing import Any

# Token-id / logprob lists dominate trace memory. As Python list[int]/list[float]
# each element costs ~32-36 B (an 8 B pointer plus a boxed int/float object) since
# vocab ids exceed CPython's small-int compact cap (256). Packed into array.array
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


# Sentinel key marking an compacted ``messages`` field: [session_id, leaf_node_id, length].
# Present only inside the store; readers always see the expanded list.
_MESSAGES_REF_KEY = "__messages_ref__"

# Sentinel for delta-stored prompt token ids: [prev_trace_id | None, lcp, suffix].
# Messages were only half the quadratic — every trace also carries the FULL
# prompt token ids of its call (up to ~128k ids), which repeat the previous
# call's ids almost entirely. Compact mode stores only the suffix beyond the
# longest common prefix with the session's previous trace; reads rebuild the
# full list by walking the chain. LCP against the actual previous ids makes
# this lossless regardless of renderer behavior: a non-extending prompt just
# gets lcp=0 and stores verbatim.
_PROMPT_IDS_DELTA_KEY = "__prompt_ids_delta__"


def _message_fingerprint(message: dict[str, Any]) -> str:
    """Fingerprint of one message (order-sensitive JSON → sha256).

    Deliberately order-SENSITIVE: equal content under different key order
    stays a distinct node, so reads reproduce exactly what was stored —
    byte-for-byte — rather than a semantically-equal reordering.
    """
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class MemoryTraceStore:
    """Ephemeral in-memory store.  Useful for tests and short-lived processes.

    ``compact=True`` (the ``compact`` store worker) opts into message
    compaction: agentic clients resend the whole conversation on every call, so
    trace N of a session carries the same N-1 leading messages as trace N-1 —
    storing each trace's ``messages`` verbatim makes a session's memory
    quadratic in turns (measured 30–400 MB per task on DeepSWE-length
    rollouts). Like ``_pack`` does for token-id lists, compact mode compacts on
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
        # re-compaction of prefixes without walking or re-storing them.
        self._session_node_lookup: dict[str, dict[tuple[str | None, str], str]] = defaultdict(dict)
        # (session_id, lineage_id) -> (trace_id, full prompt ids of that
        # lineage's latest trace) — the LCP reference for prompt-id delta
        # storage. Keyed per lineage so interleaved parent/subagent
        # conversations never delta against each other's prompts.
        self._session_last_prompt_ids: dict[tuple[str, str | None], tuple[str, list[int]]] = {}

    def _messages_to_nodes(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Return ``data`` with ``messages`` replaced by a leaf reference."""
        messages = data.get("messages")
        if type(messages) is not list or not messages or not all(isinstance(m, dict) for m in messages):
            return data  # empty/unknown shape: store verbatim
        nodes = self._session_nodes[session_id]
        chain = self._session_node_lookup[session_id]
        parent: str | None = None
        for message in messages:
            key = (parent, _message_fingerprint(message))
            node_id = chain.get(key)
            if node_id is None:
                # Merkle id: parent chain + content, so position matters.
                node_id = hashlib.sha256(f"{parent}:{key[1]}".encode()).hexdigest()
                nodes[node_id] = (parent, message)
                chain[key] = node_id
            parent = node_id
        out = dict(data)
        out["messages"] = {_MESSAGES_REF_KEY: [session_id, parent, len(messages)]}
        return out

    def _nodes_to_messages(self, data: dict[str, Any]) -> dict[str, Any]:
        """Inverse of :meth:`_messages_to_nodes` — rebuild the full message list."""
        marker = data.get("messages")
        if type(marker) is not dict or _MESSAGES_REF_KEY not in marker:
            return data
        session_id, leaf, length = marker[_MESSAGES_REF_KEY]
        nodes = self._session_nodes.get(session_id, {})
        messages: list[dict[str, Any]] = []
        node_id = leaf
        seen: set[str] = set()
        while node_id is not None:
            if node_id in seen:
                raise ValueError(f"message node cycle at {node_id!r}")
            seen.add(node_id)
            entry = nodes.get(node_id)
            if entry is None:
                raise ValueError(f"dangling message node reference {node_id!r}")
            parent, message = entry
            messages.append(message)
            node_id = parent
        messages.reverse()
        if len(messages) != length:
            raise ValueError(f"compacted chain length {len(messages)} != recorded {length}")
        out = dict(data)
        out["messages"] = messages
        return out

    def _rebase_delta_dependents(self, session_id: str, trace_id: str) -> None:
        """Materialize any trace whose delta references *trace_id* to verbatim ids.

        Called before a trace is overwritten (replay re-stores reuse trace
        ids): a dependent's delta encodes a prefix of the OLD content, so it
        must be pinned to full ids first or reconstruction would read the
        overwritten trace and could even form a cycle (t1 -> t2 -> t1).
        """
        memo: dict[str, list[int]] = {}
        # Trace ids are global (self._traces), so dependents may live in ANY
        # session — a session-scoped scan would leave other sessions' deltas
        # pointing at the new content and reconstructing garbage.
        for tid, stored in list(self._traces.items()):
            if tid == trace_id:
                continue
            marker = stored.get("prompt_token_ids")
            if type(marker) is dict and _PROMPT_IDS_DELTA_KEY in marker and marker[_PROMPT_IDS_DELTA_KEY][0] == trace_id:
                self._traces[tid] = _pack(self._delta_to_prompt_ids(tid, _unpack(stored), memo))
        # An anchor caches the previous trace's FULL ids; one pointing at the
        # overwritten trace would LCP against stale content and mint deltas
        # that reconstruct silently wrong ids. Anchors live per (session,
        # lineage) but reference GLOBAL trace ids — drop them everywhere
        # (third instance of the session-scoped-state-on-global-ids bug).
        for key, (anchor_tid, _) in list(self._session_last_prompt_ids.items()):
            if anchor_tid == trace_id:
                del self._session_last_prompt_ids[key]

    def _prompt_ids_to_delta(self, session_id: str, trace_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Store only the suffix of ``prompt_token_ids`` beyond the previous trace's."""
        ids = data.get("prompt_token_ids")
        if type(ids) is not list or not ids:
            return data
        if trace_id in self._traces:  # replay overwrite: pin dependents first, store verbatim
            self._rebase_delta_dependents(session_id, trace_id)
            self._session_last_prompt_ids[(session_id, data.get("lineage_id"))] = (trace_id, ids)
            return data
        key = (session_id, data.get("lineage_id"))
        prev = self._session_last_prompt_ids.get(key)
        out = dict(data)
        if prev is not None and prev[0] not in self._traces:
            prev = None  # anchor points at a deleted trace: never delta against it
        if prev is not None and prev[0] != trace_id:
            prev_tid, prev_ids = prev
            lcp = 0
            n = min(len(ids), len(prev_ids))
            while lcp < n and ids[lcp] == prev_ids[lcp]:
                lcp += 1
            if lcp > 0:
                out["prompt_token_ids"] = {_PROMPT_IDS_DELTA_KEY: [prev_tid, lcp, ids[lcp:]]}
        self._session_last_prompt_ids[key] = (trace_id, ids)
        return out

    def _full_prompt_ids(self, trace_id: str, memo: dict[str, list[int]]) -> list[int]:
        """Full prompt ids of *trace_id*, resolving delta chains iteratively."""
        if trace_id in memo:
            return memo[trace_id]
        # Walk back to the first non-delta ancestor, then rebuild forward.
        chain: list[str] = []
        visiting: set[str] = set()
        tid = trace_id
        while tid not in memo:
            if tid in visiting:
                raise ValueError(f"prompt-id delta cycle at trace {tid!r}")
            visiting.add(tid)
            data = self._traces.get(tid)
            marker = None if data is None else data.get("prompt_token_ids")
            if type(marker) is not dict or _PROMPT_IDS_DELTA_KEY not in marker:
                base = _unpack(data or {}).get("prompt_token_ids") or []
                memo[tid] = base if isinstance(base, list) else list(base)
                break
            chain.append(tid)
            tid = marker[_PROMPT_IDS_DELTA_KEY][0]
            if tid is None:
                memo[None] = []  # type: ignore[index]
                break
        for tid in reversed(chain):
            prev_tid, lcp, suffix = self._traces[tid]["prompt_token_ids"][_PROMPT_IDS_DELTA_KEY]
            prev_full = memo[prev_tid]
            if lcp > len(prev_full):
                raise ValueError(f"prompt-id delta lcp {lcp} exceeds ancestor length {len(prev_full)}")
            memo[tid] = prev_full[:lcp] + list(suffix)
        return memo[trace_id]

    def _delta_to_prompt_ids(self, trace_id: str, data: dict[str, Any], memo: dict[str, list[int]] | None = None) -> dict[str, Any]:
        """Inverse of :meth:`_prompt_ids_to_delta` for one trace dict."""
        marker = data.get("prompt_token_ids")
        if type(marker) is not dict or _PROMPT_IDS_DELTA_KEY not in marker:
            return data
        out = dict(data)
        out["prompt_token_ids"] = self._full_prompt_ids(trace_id, {} if memo is None else memo)
        return out

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        now = time.time()
        if self._compact:
            data = self._prompt_ids_to_delta(session_id, trace_id, self._messages_to_nodes(session_id, data))
        self._traces[trace_id] = _pack(data)
        if trace_id not in self._timestamps:
            self._timestamps[trace_id] = now
        idx = self._session_index[session_id]
        if trace_id not in idx:
            idx.append(trace_id)

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        data = self._traces.get(trace_id)
        if data is None:
            return None
        return self._delta_to_prompt_ids(trace_id, self._nodes_to_messages(_unpack(data)))

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        ids = self._session_index.get(session_id, [])
        results: list[tuple[str, dict[str, Any]]] = []
        for tid in ids:
            ts = self._timestamps.get(tid, 0.0)
            if since is not None and ts < since:
                continue
            data = self._traces.get(tid)
            if data is not None:
                results.append((tid, data))
        if limit is not None:
            results = results[:limit]
        memo: dict[str, list[int]] = {}
        return [self._delta_to_prompt_ids(tid, self._nodes_to_messages(_unpack(d)), memo) for tid, d in results]

    async def get_session_traces_compact(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Session traces in compact wire form: a node table plus leaf refs.

        Wire shape: ``{"format": "compact", "nodes": {id: {"p": parent|None,
        "m": message}}, "traces": [...]}`` where each trace carries
        ``messages_ref: [leaf_id|None, length]`` instead of ``messages``.
        Works in both store modes — a default-mode store compacts into an
        ephemeral table at read time — so fetch format is independent of how
        the store holds traces. The expanded and compact forms reconstruct
        identically (see client._expand_compact_traces and the parity tests).
        """
        ids = self._session_index.get(session_id, [])
        raw: list[tuple[str, dict[str, Any]]] = []
        for tid in ids:
            ts = self._timestamps.get(tid, 0.0)
            if since is not None and ts < since:
                continue
            data = self._traces.get(tid)
            if data is not None:
                raw.append((tid, data))
        if limit is not None:
            raw = raw[:limit]

        nodes_out: dict[str, dict[str, Any]] = {}
        included_tids = {tid for tid, _ in raw}
        ids_memo: dict[str, list[int]] = {}
        # Ephemeral compacter for traces stored verbatim (default mode).
        eph_nodes: dict[str, tuple[str | None, dict[str, Any]]] = {}
        eph_lookup: dict[tuple[str | None, str], str] = {}
        compacted = 0  # messages compacted since the last event-loop yield

        def _collect(leaf: str | None, table: dict[str, tuple[str | None, dict[str, Any]]]) -> None:
            node_id = leaf
            while node_id is not None and node_id not in nodes_out:
                parent, message = table[node_id]
                nodes_out[node_id] = {"p": parent, "m": message}
                node_id = parent

        traces_out: list[dict[str, Any]] = []
        for tid, data in raw:
            data = _unpack(data)
            marker = data.get("messages")
            if type(marker) is dict and _MESSAGES_REF_KEY in marker:
                # The marker records which session's node table owns its chain —
                # a trace_id re-stored under another session points there, not
                # at the requested session (matching _nodes_to_messages).
                marker_sid, leaf, length = marker[_MESSAGES_REF_KEY]
                _collect(leaf, self._session_nodes.get(marker_sid, {}))
            else:
                messages = marker
                if type(messages) is not list or not all(isinstance(m, dict) for m in (messages or [])):
                    # unknown shape: ship verbatim, no ref
                    traces_out.append(data)
                    continue
                parent: str | None = None
                for message in messages:
                    key = (parent, _message_fingerprint(message))
                    node_id = eph_lookup.get(key)
                    if node_id is None:
                        node_id = hashlib.sha256(f"{parent}:{key[1]}".encode()).hexdigest()
                        eph_nodes[node_id] = (parent, message)
                        eph_lookup[key] = node_id
                    parent = node_id
                    # Fingerprinting a long session is hundreds of MB of hashing;
                    # yield periodically so in-flight proxying is never starved.
                    compacted += 1
                    if compacted % 512 == 0:
                        await asyncio.sleep(0)
                leaf, length = parent, len(messages)
                _collect(leaf, eph_nodes)
            out = dict(data)
            out.pop("messages", None)
            out["messages_ref"] = [leaf, length]
            out["_tid"] = tid  # chain anchor for prompt_ids_delta; client strips it
            ids_marker = out.get("prompt_token_ids")
            if type(ids_marker) is dict and _PROMPT_IDS_DELTA_KEY in ids_marker:
                prev_tid = ids_marker[_PROMPT_IDS_DELTA_KEY][0]
                if prev_tid in included_tids:
                    # Ship the delta; the client rebuilds chains in order
                    # (prev refs point to an earlier trace in the payload).
                    out.pop("prompt_token_ids", None)
                    out["prompt_ids_delta"] = list(ids_marker[_PROMPT_IDS_DELTA_KEY])
                else:
                    # since/limit excluded the ancestor: rebase to full ids so
                    # the payload is self-contained.
                    out["prompt_token_ids"] = self._full_prompt_ids(tid, ids_memo)
            traces_out.append(out)

        return {"format": "compact", "nodes": nodes_out, "traces": traces_out}

    async def delete_session(self, session_id: str) -> int:
        ids = self._session_index.pop(session_id, [])
        # Collect trace_ids referenced by other sessions
        referenced: set[str] = set()
        for sid, tids in self._session_index.items():
            referenced.update(tids)
        # Traces surviving via another session may hold markers into the
        # tables about to be dropped (a trace_id re-stored under two sessions
        # points at whichever session stored it last). Re-materialize them to
        # verbatim form first, or they would be unreadable afterwards.
        memo: dict[str, list[int]] = {}
        for tid in ids:
            if tid in referenced and tid in self._traces:
                self._traces[tid] = _pack(self._delta_to_prompt_ids(tid, self._nodes_to_messages(_unpack(self._traces[tid])), memo))
        self._session_nodes.pop(session_id, None)
        self._session_node_lookup.pop(session_id, None)
        for key in [k for k in self._session_last_prompt_ids if k[0] == session_id]:
            del self._session_last_prompt_ids[key]
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
