"""Unit tests for ``derive_termination_reason``.

On the CLI-harness path the harness itself sets TIMEOUT (wall-clock budget) and
ERROR (sandbox/exec failure); anything that reaches the engine's fallback is a
clean exit → ENV_DONE. We deliberately do NOT infer MAX_RESPONSE_LENGTH_EXCEEDED
or MAX_TURNS_EXCEEDED from gateway traces here — the in-sandbox agent recovers
from a truncated response (so a trailing ``finish_reason == "length"`` is
coincidental, not terminal) and the trace count is LLM calls, not turns. These
tests pin that behavior so the misleading branches don't get re-added.
"""

from __future__ import annotations

from rllm.engine.agentflow_engine import derive_termination_reason
from rllm.engine.rollout import ModelOutput
from rllm.types import Episode, Step, TerminationReason, Trajectory


def _episode(finish_reasons: list[str | None]) -> Episode:
    steps = [Step(model_output=ModelOutput(finish_reason=fr)) for fr in finish_reasons]
    return Episode(trajectories=[Trajectory(name="t", steps=steps)])


def test_clean_exit_maps_to_env_done():
    assert derive_termination_reason(_episode(["stop"])) == TerminationReason.ENV_DONE


def test_trailing_length_finish_reason_is_not_treated_as_terminal():
    # Terminus-2 recovers from truncation and keeps going, so a "length" on the
    # final trace is not the reason the episode ended. Classifying it would, with
    # base.yaml's mask, silently drop otherwise-fine (incl. successful) episodes.
    assert derive_termination_reason(_episode(["stop", "length"])) == TerminationReason.ENV_DONE


def test_empty_episode_maps_to_env_done():
    assert derive_termination_reason(Episode(trajectories=[])) == TerminationReason.ENV_DONE
