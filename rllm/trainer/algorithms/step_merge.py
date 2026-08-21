"""Step-sequence helpers shared by the token-level backend transforms.

A multi-turn trajectory is merged into as few training sequences as possible: when
each step's prompt extends the previous step's prompt+response, the whole turn
chain is one sequence. These two helpers decide where those chains start and stop.
Kept free of any backend SDK so every transform can use them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["is_prefix", "partition_steps_by_lineage"]


def is_prefix(seq1: list[Any], seq2: list[Any]) -> bool:
    """Whether ``seq1`` is a prefix of ``seq2``."""
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def partition_steps_by_lineage(steps: list[Any]) -> list[list[Any]]:
    """Group steps by gateway ``lineage_id`` (from ``step.metadata``), in first-appearance order.

    A subagent runs under the same session with its own system prompt, so its turns
    are not prefix-extensions of the parent's. Partitioning first means each lineage
    merges independently instead of fragmenting into one sequence per turn, even when
    parent and subagent turns interleave in time.

    Untagged steps (cumulative mode off, or eval) share the ``None`` key and so form a
    single partition — the un-partitioned behavior.
    """
    groups: dict[Any, list[Any]] = {}
    order: list[Any] = []
    for step in steps:
        lid = (step.metadata or {}).get("lineage_id")
        if lid not in groups:
            groups[lid] = []
            order.append(lid)
        groups[lid].append(step)
    return [groups[lid] for lid in order]
