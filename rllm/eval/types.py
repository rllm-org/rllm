"""Eval result data shapes.

:class:`EvalOutput` and :class:`Signal` are the values produced by an
:class:`~rllm.types.Evaluator`. They live here because they describe
scoring outputs; the producer/consumer protocols themselves are core
types in :mod:`rllm.types`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Signal:
    """A single named evaluation signal."""

    name: str  # e.g. "accuracy", "format", "f1"
    value: float  # typically 0.0-1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalOutput:
    """Evaluation result for one example."""

    reward: float
    is_correct: bool
    signals: list[Signal] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
