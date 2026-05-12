"""Tests for backend-neutral DPO preference-pair construction."""

from __future__ import annotations

import pytest

from rllm.experimental.common.preference import DPOConfig, DPOPairingStrategy, PreferencePair, build_preference_pairs
from rllm.experimental.unified_trainer import TrainerState
from rllm.types import Step, Trajectory, TrajectoryGroup


def _make_trajectory(
    *,
    reward: float,
    prompt_ids: list[int],
    response_ids: list[int],
    n_steps: int = 1,
) -> Trajectory:
    steps = []
    for step_idx in range(n_steps):
        steps.append(
            Step(
                prompt_ids=prompt_ids if step_idx == 0 else [*prompt_ids, step_idx],
                response_ids=response_ids,
                reward=reward,
            )
        )
    return Trajectory(steps=steps, reward=reward)


def _make_group(*trajectories: Trajectory, group_id: str = "task:solver") -> TrajectoryGroup:
    return TrajectoryGroup(trajectories=list(trajectories), group_id=group_id)


class TestDPOConfig:
    def test_normalizes_pairing_strategy(self):
        cfg = DPOConfig(pairing_strategy="best_worst")
        assert cfg.pairing_strategy == DPOPairingStrategy.BEST_WORST

    def test_rejects_non_positive_beta(self):
        with pytest.raises(ValueError, match="beta"):
            DPOConfig(beta=0.0)

    def test_rejects_negative_reward_gap(self):
        with pytest.raises(ValueError, match="min_reward_gap"):
            DPOConfig(min_reward_gap=-0.1)


class TestBuildPreferencePairs:
    def test_builds_best_worst_pair(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=0.1, prompt_ids=[1, 2], response_ids=[11]),
            group_id="task:solver",
        )

        pairs, metrics = build_preference_pairs([group], DPOConfig())

        assert len(pairs) == 1
        pair = pairs[0]
        assert isinstance(pair, PreferencePair)
        assert pair.group_id == "task:solver"
        assert pair.task_id == "task"
        assert pair.role == "solver"
        assert pair.reward_gap == pytest.approx(0.9)
        assert metrics["dpo/pairs_built"] == 1
        assert metrics["dpo/reward_gap_mean"] == pytest.approx(0.9)

    def test_skips_prompt_mismatch(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=0.0, prompt_ids=[9, 9], response_ids=[11]),
        )

        pairs, metrics = build_preference_pairs([group], DPOConfig())

        assert pairs == []
        assert metrics["dpo/groups_skipped_prompt_mismatch"] == 1
        assert metrics["dpo/pairs_built"] == 0

    def test_skips_multistep_groups(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10], n_steps=2),
            _make_trajectory(reward=0.0, prompt_ids=[1, 2], response_ids=[11]),
        )

        pairs, metrics = build_preference_pairs([group], DPOConfig())

        assert pairs == []
        assert metrics["dpo/groups_skipped_multistep"] == 1

    def test_skips_ties_by_default(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[11]),
        )

        pairs, metrics = build_preference_pairs([group], DPOConfig())

        assert pairs == []
        assert metrics["dpo/groups_skipped_tie"] == 1

    def test_can_keep_ties_when_configured(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[11]),
        )

        pairs, metrics = build_preference_pairs([group], DPOConfig(drop_ties=False))

        assert len(pairs) == 1
        assert pairs[0].reward_gap == 0.0
        assert metrics["dpo/pairs_built"] == 1

    def test_skips_small_reward_gap(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=0.95, prompt_ids=[1, 2], response_ids=[11]),
        )

        pairs, metrics = build_preference_pairs([group], DPOConfig(min_reward_gap=0.1))

        assert pairs == []
        assert metrics["dpo/groups_skipped_small_gap"] == 1


def test_trainer_state_has_no_generic_preference_pair_slot():
    trainer_state = TrainerState()
    assert not hasattr(trainer_state, "preference_pairs")
