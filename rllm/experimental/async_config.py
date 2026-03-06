"""Configuration for async (concurrent generation + training) mode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AsyncTrainingConfig:
    """Controls the async training behavior spectrum.

    When `enabled` is False, the trainer uses the current synchronous pipeline.
    When `enabled` is True, the trainer runs concurrent generation + training
    with staleness-based filtering.

    Behavior spectrum:
        - max_staleness=0, buffer_size=1: Effectively synchronous (backpressure serializes)
        - max_staleness=1, buffer_size=2: 1-step overlap
        - max_staleness=k, buffer_size=k+1: k-step off-policy
        - max_staleness=5, buffer_size=8: Fully async with aggressive filtering
    """

    enabled: bool = False
    max_staleness: int = 0
    buffer_size: int = 1
    requeue_stale: bool = True
