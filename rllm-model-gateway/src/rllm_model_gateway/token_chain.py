"""Rebuild full ``prompt_token_ids`` from delta-chain trace storage.

In cumulative_token_mode the gateway stores each turn's prompt as a *delta* (the
newly rendered suffix beyond the previous turn) plus a ``parent_trace_id`` pointer,
instead of the full cumulative prompt (which grows quadratically over a session —
see ``design/gateway-dag-token-storage.md``). This module walks that chain to
reconstruct the full ``prompt_token_ids`` on read, so HTTP clients and the trainer
see the exact same ``TraceRecord`` shape as before delta storage existed.

A trace is either:
- a **root** — ``prompt_delta_token_ids is None``; ``prompt_token_ids`` already holds
  the full prompt (turn 0, post-reset, a duplicate replay, or a legacy/non-cumulative
  trace); or
- a **link** — ``prompt_delta_token_ids`` is the suffix and ``parent_trace_id`` names
  the predecessor. Full prompt = parent's cumulative (prompt+completion, itself
  reconstructed) + this delta.

Reconstruction is by parent pointer (not list position), so it is correct regardless
of the order traces arrive in. The training path fetches a whole session, so every
chain resolves; a partial query (``since``/``limit``) may omit a link's ancestor —
such a link is left **untouched** (still a delta) rather than reconstructed wrongly.
"""

from __future__ import annotations

from typing import Any


def reconstruct_prompt_ids(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill each resolvable trace's full ``prompt_token_ids`` from the delta chain, in place.

    For every link whose ancestry is fully present, sets ``prompt_token_ids`` to the
    reconstructed full prompt and clears ``prompt_delta_token_ids`` (so the payload
    carries the full prompt once, not the full prompt *and* the delta). Root traces,
    and links whose ancestry is incomplete in this batch, are left unchanged. Returns
    the same list for convenience.
    """
    by_id: dict[str, dict[str, Any]] = {t["trace_id"]: t for t in traces if t.get("trace_id")}
    full_cache: dict[str, list[int] | None] = {}  # trace_id -> full prompt ids, or None if unresolvable
    cum_cache: dict[str, list[int] | None] = {}  # trace_id -> prompt+completion, or None

    def _full_prompt(tid: str) -> list[int] | None:
        if tid in full_cache:
            return full_cache[tid]
        # Walk up to the deepest uncomputed ancestor, then fold back down — iterative
        # to stay O(depth) without Python recursion limits on long chains.
        path: list[str] = []
        cur: str | None = tid
        seen: set[str] = set()
        while cur is not None and cur not in full_cache:
            node = by_id.get(cur)
            if node is None or cur in seen:
                full_cache[cur] = None  # missing node or cycle → unresolvable
                break
            seen.add(cur)
            if node.get("prompt_delta_token_ids") is None:
                full_cache[cur] = list(node.get("prompt_token_ids") or [])  # root
                break
            parent_id = node.get("parent_trace_id")
            if not parent_id or parent_id not in by_id:
                full_cache[cur] = None  # link with an absent parent → unresolvable
                break
            path.append(cur)
            cur = parent_id
        for node_id in reversed(path):
            node = by_id[node_id]
            base = _cumulative(node.get("parent_trace_id"))
            full_cache[node_id] = None if base is None else base + list(node.get("prompt_delta_token_ids") or [])
        return full_cache.get(tid)

    def _cumulative(tid: str | None) -> list[int] | None:
        if tid is None:
            return []
        if tid in cum_cache:
            return cum_cache[tid]
        fp = _full_prompt(tid)
        node = by_id.get(tid) or {}
        cum = None if fp is None else fp + list(node.get("completion_token_ids") or [])
        cum_cache[tid] = cum
        return cum

    for t in traces:
        if t.get("prompt_delta_token_ids") is None:
            continue  # root / legacy: prompt_token_ids already full
        tid = t.get("trace_id")
        if not tid:
            continue
        full = _full_prompt(tid)
        if full is not None:
            t["prompt_token_ids"] = full
            t["prompt_delta_token_ids"] = None
    return traces
