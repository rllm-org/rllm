"""Lossless compact converter for saved episode JSON (episode compact format).

Eval episodes store every step's full ``chat_completions`` snapshot, so a
saved episode repeats each message once per subsequent step — quadratic in
steps (measured 9.3 GB for one 113-task DeepSWE run, 396 MB for a single
229-step episode). The compact format stores each unique message once per trajectory in
a Merkle-chained node table and replaces per-step lists with a leaf
reference; :func:`expand_episode` restores legacy-format exactly.

The invariant is strict losslessness: ``expand_episode(compact_episode(x))``
equals ``x`` — verified canonical-byte-for-byte against every real eval dump
in ``tests/eval/test_episode_compact.py``, including non-cumulative
histories (rewrites fork the chain) and repeated identical content at
different positions (node identity hashes the parent chain plus content, so
positions never collapse).

Wire/store cousins: the same node-table shape backs the gateway's compact
trace store and ``?format=compact`` fetch (rllm-model-gateway). This module
is dependency-free of both so episode files stand alone.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Episode-level format tag. Absent means the legacy verbatim format.
FORMAT_KEY = "episode_format"
COMPACT_FORMAT = "compact"
# Encoding version, stamped alongside the tag. Bumped whenever node identity
# or marker shape changes, so recompression differences across versions are
# detectable instead of silent (review: the fingerprint scheme changed during
# development while the tag stayed constant).
VERSION_KEY = "compact_version"
COMPACT_VERSION = 1

_NODES_KEY = "message_nodes"
# In compact form a step keeps its "chat_completions" KEY but the value is a
# marker dict {"messages_ref": [leaf, length]} — a value swap, not a key swap,
# so expanding restores the file byte-for-byte including dict key order.
_REF_MARKER = "messages_ref"


def _message_fp(message: dict[str, Any]) -> str:
    # Deliberately order-SENSITIVE (no sort_keys): two messages with equal
    # content but different key order must stay distinct nodes, or expansion
    # would return the first-seen dict for both and the file would no longer
    # be byte-identical after a round trip (found by file-byte parity against
    # real dumps that mix role-first and content-first messages).
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def compact_episode(data: dict[str, Any]) -> dict[str, Any]:
    """Return the compact-format form of a legacy-format episode dict (idempotent)."""
    if data.get(FORMAT_KEY) == COMPACT_FORMAT:
        return data
    if FORMAT_KEY in data or VERSION_KEY in data:
        return data  # reserved keys already in use: refuse, stay legacy (lossless)
    out = dict(data)
    out_trajectories = []
    for trajectory in data.get("trajectories") or []:
        if _NODES_KEY in trajectory:
            # reserved key already present: keep this trajectory verbatim
            out_trajectories.append(trajectory)
            continue
        steps = trajectory.get("steps") or []
        marked = 0
        nodes: dict[str, dict[str, Any]] = {}
        chain: dict[tuple[str | None, str], str] = {}
        out_steps = []
        compactable = True
        for step in steps:
            cc = step.get("chat_completions")
            if type(cc) is not list or not all(isinstance(m, dict) for m in cc):
                compactable = False
                break
            parent: str | None = None
            for message in cc:
                key = (parent, _message_fp(message))
                node_id = chain.get(key)
                if node_id is None:
                    node_id = hashlib.sha256(f"{parent}:{key[1]}".encode()).hexdigest()
                    nodes[node_id] = {"p": parent, "m": message}
                    chain[key] = node_id
                parent = node_id
            marker = {_REF_MARKER: [parent, len(cc)]}
            marked += 1
            out_step = {k: (marker if k == "chat_completions" else v) for k, v in step.items()}
            out_steps.append(out_step)
        if not compactable or marked == 0:
            # Unknown step shape, or nothing to compact (e.g. zero steps):
            # keep this trajectory verbatim — expand passes it through, so no
            # empty node table may be introduced (symmetry, byte-exactness).
            out_trajectories.append(trajectory)
            continue
        out_trajectory = dict(trajectory)
        out_trajectory["steps"] = out_steps
        out_trajectory[_NODES_KEY] = nodes
        out_trajectories.append(out_trajectory)
    if "trajectories" in data:
        out["trajectories"] = out_trajectories
    out[FORMAT_KEY] = COMPACT_FORMAT
    out[VERSION_KEY] = COMPACT_VERSION
    return out


def expand_episode(data: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`compact_episode`; identity on legacy-format dicts."""
    if data.get(FORMAT_KEY) != COMPACT_FORMAT:
        return data
    version = data.get(VERSION_KEY)
    if version is not None and version != COMPACT_VERSION:
        raise ValueError(f"unsupported {VERSION_KEY} {version!r}; this build reads <= {COMPACT_VERSION}")
    prerelease = version is None  # unversioned = pre-release local artifacts
    out = dict(data)
    out.pop(FORMAT_KEY)
    out.pop(VERSION_KEY, None)
    out_trajectories = []
    for trajectory in data.get("trajectories") or []:
        nodes = trajectory.get(_NODES_KEY)
        if nodes is None:
            out_trajectories.append(trajectory)
            continue
        # leaf -> materialized list; shared prefixes walk once, share objects.
        paths: dict[str | None, list[dict[str, Any]]] = {None: []}

        def _path(leaf: str | None, *, nodes=nodes, paths=paths) -> list[dict[str, Any]]:
            # Cache only REQUESTED leaves: caching every intermediate prefix
            # made one deep chain cost O(M^2) allocations (review); extending
            # once from the nearest cached ancestor keeps shared-prefix reuse
            # while a lone leaf costs O(M).
            if leaf in paths:
                return paths[leaf]
            todo: list[str] = []
            seen: set[str] = set()
            node_id = leaf
            while node_id is not None and node_id not in paths:
                if node_id in seen:
                    raise ValueError(f"message node cycle at {node_id!r}")
                seen.add(node_id)
                todo.append(node_id)
                node = nodes.get(node_id)
                if node is None:
                    raise ValueError(f"dangling message node reference {node_id!r}")
                node_id = node["p"]
            base = list(paths[node_id if node_id is not None else None])
            base.extend(nodes[nid]["m"] for nid in reversed(todo))
            paths[leaf] = base
            return base

        out_steps = []
        resolved_any = False
        for step in trajectory.get("steps") or []:
            marker = step.get("chat_completions")
            if prerelease and (type(marker) is not dict or _REF_MARKER not in marker) and type(step.get("chat_completions_ref")) is list:
                # pre-release artifact (early development head): the ref lived
                # under its own key instead of swapping the value in place.
                marker = {_REF_MARKER: step["chat_completions_ref"]}
                step = {k: v for k, v in step.items() if k != "chat_completions_ref"}
            if type(marker) is not dict or _REF_MARKER not in marker:
                out_steps.append(step)
                continue
            resolved_any = True
            leaf, length = marker[_REF_MARKER]
            messages = _path(leaf)
            if len(messages) != length:
                raise ValueError(f"compact chain length {len(messages)} != recorded {length}")
            if "chat_completions" in step:
                out_step = {k: (messages if k == "chat_completions" else v) for k, v in step.items()}
            else:  # pre-release artifact: the key was removed, not value-swapped
                out_step = dict(step)
                out_step["chat_completions"] = messages
            out_steps.append(out_step)
        if not resolved_any:
            if prerelease and nodes == {}:
                # Pre-release writers stamped empty node tables onto zero-step
                # trajectories; strip them so the export is true legacy.
                out_trajectory = {k: v for k, v in trajectory.items() if k != _NODES_KEY}
                out_trajectories.append(out_trajectory)
                continue
            # A trajectory can carry a user "message_nodes" key without any
            # compact markers (compaction refuses those) — pass it through.
            out_trajectories.append(trajectory)
            continue
        out_trajectory = dict(trajectory)
        out_trajectory.pop(_NODES_KEY)
        out_trajectory["steps"] = out_steps
        out_trajectories.append(out_trajectory)
    if "trajectories" in data:
        out["trajectories"] = out_trajectories
    return out
