"""Backend-agnostic step merging for cumulative-prefix trajectories.

Multi-turn trajectories whose successive steps' prompts each extend the
previous step's full sequence (a ReAct/tool-call agent appending tool
results and assistant turns) collapse into a SINGLE training row whose
response is ``[A0, obs1, A1, obs2, A2, ...]`` with ``response_mask`` 1 on
action tokens and 0 on observation tokens. A non-extending step opens a
new segment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rllm.experimental.rollout.types import TokenInput
from rllm.types import Step, Trajectory


class TokenOps(Protocol):
    """Backend-typed token-list ops the merge depends on.

    Verl-style backends (``list[int]`` prompts) can use the default
    :class:`DefaultTokenOps`. Backends with non-int chunk elements (Tinker)
    ship their own ops alongside their transform.
    """

    def flatten_prompt(self, prompt: TokenInput) -> TokenInput:
        """Return a comparable / storable form of ``prompt``."""
        ...

    def flat_token_length(self, token_input: TokenInput) -> int:
        """Token count; chunk elements contribute their ``.length``."""
        ...


@dataclass(frozen=True)
class DefaultTokenOps:
    """Default ops for ``TokenInput`` token sequences."""

    def flatten_prompt(self, prompt: list[int]) -> list[int]:
        return list(prompt)

    def flat_token_length(self, token_input: list[int]) -> int:
        return len(token_input)


_DEFAULT_TOKEN_OPS = DefaultTokenOps()


@dataclass
class MergedSegment:
    """One row from prefix-merging a trajectory's steps.

    The per-token arrays (``response_mask``, ``response_logprobs``,
    ``response_advantages``, and per-token entries in ``extras``) are sized
    to the FLAT token count of ``response_ids`` — for Verl this equals
    ``len(response_ids)``; for Tinker, non-int chunks contribute their
    ``.length``.
    """

    prompt_ids: TokenInput
    response_ids: TokenInput
    response_mask: list[int]
    response_logprobs: list[float]
    response_advantages: list[float]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_response_tokens(self) -> int:
        return len(self.response_mask)


@dataclass(frozen=True)
class PerTokenExtras:
    """Spec for a per-token extras stream attached to each segment.

    ``extractor`` returns a list aligned with ``len(step.response_ids)``,
    or ``None`` if the step lacks the field. Tokens introduced by
    delta-observation regions, and steps whose extractor returned ``None``,
    are filled with ``pad_value``.
    """

    extractor: Callable[[Step], Sequence[Any] | None]
    pad_value: Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_advantage(adv: float | list[float] | None, n: int, *, require: bool) -> list[float]:
    if adv is None:
        if require:
            raise AssertionError("step.advantage is None")
        return [0.0] * n
    if isinstance(adv, list):
        if len(adv) != n:
            raise AssertionError(f"step.advantage length {len(adv)} != response length {n}")
        return [float(x) for x in adv]
    return [float(adv)] * n


def _normalize_logprobs(lp: Sequence[float] | None, n: int, *, require: bool, pad: bool) -> list[float]:
    if not lp:
        if require:
            raise AssertionError("step.logprobs is empty")
        return []
    out = list(lp)
    if len(out) == n:
        return out
    if len(out) < n and pad:
        return out + [0.0] * (n - len(out))
    raise AssertionError(f"step.logprobs length {len(out)} != response length {n}")


def _step_has_model_output(step: Step) -> bool:
    return step.model_output is not None and getattr(step.model_output, "prompt_ids", None) is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_trajectory_steps(
    trajectory: Trajectory,
    *,
    token_ops: TokenOps = _DEFAULT_TOKEN_OPS,
    require_logprobs: bool = False,
    require_advantage: bool = False,
    pad_short_logprobs: bool = False,
    skip_steps_without_model_output: bool = False,
    per_token_extras: dict[str, PerTokenExtras] | None = None,
    per_segment_extras: dict[str, Callable[[Step], Any]] | None = None,
) -> list[MergedSegment]:
    """Walk ``trajectory.steps`` and emit one segment per cumulative-prefix run.

    Tinker callers pass a chunk-aware ``token_ops`` and set
    ``require_logprobs=True``, ``require_advantage=True``. Verl callers use
    the default ``DefaultTokenOps`` and set ``pad_short_logprobs=True``,
    ``skip_steps_without_model_output=True``.
    """
    if not trajectory.steps:
        return []

    valid_steps: list[Step] = [s for s in trajectory.steps if not (skip_steps_without_model_output and not _step_has_model_output(s))]
    if not valid_steps:
        return []

    per_token_extras = per_token_extras or {}
    per_segment_extras = per_segment_extras or {}

    def _start(step: Step) -> tuple[MergedSegment, list[Any]]:
        prompt_flat = token_ops.flatten_prompt(step.prompt_ids or [])
        action = list(step.response_ids or [])
        n = len(action)

        logprobs = _normalize_logprobs(step.logprobs, n, require=require_logprobs, pad=pad_short_logprobs)
        advantages = _normalize_advantage(step.advantage, n, require=require_advantage)

        extras: dict[str, Any] = {}
        for name, spec in per_token_extras.items():
            raw = spec.extractor(step)
            extras[name] = list(raw) if raw and len(raw) == n else [spec.pad_value] * n
        for name, extractor in per_segment_extras.items():
            extras[name] = extractor(step)

        seg = MergedSegment(
            prompt_ids=list(prompt_flat),
            response_ids=list(action),
            response_mask=[1] * n,
            response_logprobs=list(logprobs),
            response_advantages=list(advantages),
            extras=extras,
        )
        return seg, list(prompt_flat) + list(action)

    def _extend(seg: MergedSegment, full: list[Any], step: Step, delta: list[Any]) -> list[Any]:
        action = list(step.response_ids or [])
        n = len(action)
        delta_n = token_ops.flat_token_length(delta)

        logprobs = _normalize_logprobs(step.logprobs, n, require=require_logprobs, pad=pad_short_logprobs)
        advantages = _normalize_advantage(step.advantage, n, require=require_advantage)

        seg.response_ids.extend(delta)
        seg.response_ids.extend(action)
        seg.response_mask.extend([0] * delta_n + [1] * n)

        # Keep response_logprobs empty until *some* step in this segment
        # contributes — the caller uses emptiness as "no rollout-logprobs
        # in this segment, skip the field". Once non-empty, keep it
        # length-aligned (backfill earlier action regions with zeros).
        if seg.response_logprobs or logprobs:
            if not seg.response_logprobs:
                seg.response_logprobs = [0.0] * (len(seg.response_mask) - delta_n - n)
            seg.response_logprobs.extend([0.0] * delta_n)
            seg.response_logprobs.extend(logprobs if logprobs else [0.0] * n)

        seg.response_advantages.extend([0.0] * delta_n + advantages)

        for name, spec in per_token_extras.items():
            seg.extras[name].extend([spec.pad_value] * delta_n)
            raw = spec.extractor(step)
            seg.extras[name].extend(list(raw) if raw and len(raw) == n else [spec.pad_value] * n)

        full.extend(delta)
        full.extend(action)
        return full

    segments: list[MergedSegment] = []
    seg, full = _start(valid_steps[0])
    for step in valid_steps[1:]:
        prompt_flat = token_ops.flatten_prompt(step.prompt_ids or [])
        if len(prompt_flat) >= len(full) and list(prompt_flat[: len(full)]) == list(full):
            delta = list(prompt_flat[len(full) :])
            full = _extend(seg, full, step, delta)
        else:
            segments.append(seg)
            seg, full = _start(step)
    segments.append(seg)
    return segments


__all__ = [
    "DefaultTokenOps",
    "MergedSegment",
    "PerTokenExtras",
    "TokenOps",
    "merge_trajectory_steps",
]
