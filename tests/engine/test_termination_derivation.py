"""Unit tests for ``derive_termination_reason``.

The CLI-harness path returns an outcome Episode with ``termination_reason=None``
on a clean exit; the engine then classifies *why* it ended from the enriched
gateway Steps (last call's ``finish_reason``) and the harness's declared turn
cap. This is what makes compact filtering work on the SandboxedAgentFlow path
(everything used to come back ENV_DONE).
"""

from __future__ import annotations

from rllm.engine.agentflow_engine import derive_termination_reason
from rllm.engine.rollout import ModelOutput
from rllm.types import Episode, Step, TerminationReason, Trajectory


def _episode(finish_reasons: list[str | None]) -> Episode:
    steps = [Step(model_output=ModelOutput(finish_reason=fr)) for fr in finish_reasons]
    return Episode(trajectories=[Trajectory(name="t", steps=steps)])


def test_length_finish_reason_maps_to_max_response_length():
    # ...and wins even when the turn cap is also reached.
    ep = _episode(["stop", "length"])
    assert derive_termination_reason(ep, max_turns=2) == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED


def test_step_count_at_cap_maps_to_max_turns():
    ep = _episode(["stop", "stop", "stop"])
    assert derive_termination_reason(ep, max_turns=3) == TerminationReason.MAX_TURNS_EXCEEDED


def test_under_cap_maps_to_env_done():
    ep = _episode(["stop", "stop", "stop"])
    assert derive_termination_reason(ep, max_turns=10) == TerminationReason.ENV_DONE


def test_no_cap_clean_exit_maps_to_env_done():
    ep = _episode(["tool_calls", "stop"])
    assert derive_termination_reason(ep, max_turns=None) == TerminationReason.ENV_DONE


def test_empty_episode_maps_to_env_done():
    assert derive_termination_reason(Episode(trajectories=[]), max_turns=5) == TerminationReason.ENV_DONE


def test_missing_model_output_does_not_crash():
    ep = Episode(trajectories=[Trajectory(name="t", steps=[Step(model_output=None)])])
    assert derive_termination_reason(ep, max_turns=None) == TerminationReason.ENV_DONE
