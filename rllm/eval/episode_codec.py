"""Lossless compact codec for saved episode JSON (episode schema 2).

Eval episodes store every step's full ``chat_completions`` snapshot, so a
saved episode repeats each message once per subsequent step — quadratic in
steps (measured 9.3 GB for one 113-task DeepSWE run, 396 MB for a single
229-step episode). Schema 2 stores each unique message once per trajectory in
a Merkle-chained node table and replaces per-step lists with a leaf
reference; :func:`expand_episode` restores schema-1 exactly.

The invariant is strict losslessness: ``expand_episode(compact_episode(x))``
equals ``x`` — verified canonical-byte-for-byte against every real eval dump
in ``tests/eval/test_episode_codec.py``, including non-cumulative
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

# Episode-level schema tag. Absent or != 2 means legacy verbatim schema 1.
SCHEMA_KEY = "episode_schema"
COMPACT_SCHEMA = 2

_NODES_KEY = "message_nodes"
_REF_KEY = "chat_completions_ref"


def _message_fp(message: dict[str, Any]) -> str:
    payload = json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def compact_episode(data: dict[str, Any]) -> dict[str, Any]:
    """Return the schema-2 form of a schema-1 episode dict (idempotent)."""
    if data.get(SCHEMA_KEY) == COMPACT_SCHEMA:
        return data
    out = dict(data)
    out_trajectories = []
    for trajectory in data.get("trajectories") or []:
        steps = trajectory.get("steps") or []
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
            out_step = dict(step)
            del out_step["chat_completions"]
            out_step[_REF_KEY] = [parent, len(cc)]
            out_steps.append(out_step)
        if not compactable:
            # Unknown step shape: keep this trajectory verbatim (still schema-2
            # overall; expand treats trajectories without a node table as-is).
            out_trajectories.append(trajectory)
            continue
        out_trajectory = dict(trajectory)
        out_trajectory["steps"] = out_steps
        out_trajectory[_NODES_KEY] = nodes
        out_trajectories.append(out_trajectory)
    if "trajectories" in data:
        out["trajectories"] = out_trajectories
    out[SCHEMA_KEY] = COMPACT_SCHEMA
    return out


def expand_episode(data: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`compact_episode`; identity on schema-1 dicts."""
    if data.get(SCHEMA_KEY) != COMPACT_SCHEMA:
        return data
    out = dict(data)
    out.pop(SCHEMA_KEY)
    out_trajectories = []
    for trajectory in data.get("trajectories") or []:
        nodes = trajectory.get(_NODES_KEY)
        if nodes is None:
            out_trajectories.append(trajectory)
            continue
        # leaf -> materialized list; shared prefixes walk once, share objects.
        paths: dict[str | None, list[dict[str, Any]]] = {None: []}

        def _path(leaf: str | None) -> list[dict[str, Any]]:
            if leaf in paths:
                return paths[leaf]
            todo: list[str] = []
            node_id = leaf
            while node_id is not None and node_id not in paths:
                todo.append(node_id)
                node_id = nodes[node_id]["p"]
            base = paths[node_id if node_id is not None else None]
            for nid in reversed(todo):
                base = base + [nodes[nid]["m"]]
                paths[nid] = base
            return paths[leaf]

        out_steps = []
        for step in trajectory.get("steps") or []:
            ref = step.get(_REF_KEY)
            if ref is None:
                out_steps.append(step)
                continue
            leaf, length = ref
            messages = _path(leaf)
            if len(messages) != length:
                raise ValueError(f"compact chain length {len(messages)} != recorded {length}")
            out_step = dict(step)
            del out_step[_REF_KEY]
            out_step["chat_completions"] = messages
            out_steps.append(out_step)
        out_trajectory = dict(trajectory)
        out_trajectory.pop(_NODES_KEY)
        out_trajectory["steps"] = out_steps
        out_trajectories.append(out_trajectory)
    if "trajectories" in data:
        out["trajectories"] = out_trajectories
    return out
