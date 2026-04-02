"""Tests for strict DPO pair construction and trainer branching."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from rllm.experimental.common.config import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TrainingObjective,
    TransformConfig,
)
from rllm.experimental.common.preference import PreferencePair, build_preference_pairs
from rllm.experimental.common.transform import _default_traj_grouping_hook
from rllm.experimental.unified_trainer import TrainerState, UnifiedTrainer
from rllm.types import Episode, Step, Trajectory, TrajectoryGroup


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


def _make_episode(episode_id: str, reward: float, prompt_ids: list[int], response_ids: list[int]) -> Episode:
    return Episode(id=episode_id, trajectories=[_make_trajectory(reward=reward, prompt_ids=prompt_ids, response_ids=response_ids)], is_correct=reward > 0)


class TestBuildPreferencePairs:
    def test_builds_best_worst_pair(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=0.1, prompt_ids=[1, 2], response_ids=[11]),
            group_id="task:solver",
        )

        pairs, metrics = build_preference_pairs([group], AlgorithmConfig(objective="dpo").dpo)

        assert len(pairs) == 1
        pair = pairs[0]
        assert isinstance(pair, PreferencePair)
        assert pair.group_id == "task:solver"
        assert pair.task_id == "task"
        assert pair.role == "solver"
        assert pair.reward_gap == 0.9
        assert metrics["dpo/pairs_built"] == 1
        assert metrics["dpo/reward_gap_mean"] == 0.9

    def test_skips_prompt_mismatch(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=0.0, prompt_ids=[9, 9], response_ids=[11]),
        )

        pairs, metrics = build_preference_pairs([group], AlgorithmConfig(objective="dpo").dpo)

        assert pairs == []
        assert metrics["dpo/groups_skipped_prompt_mismatch"] == 1
        assert metrics["dpo/pairs_built"] == 0

    def test_skips_multistep_groups(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10], n_steps=2),
            _make_trajectory(reward=0.0, prompt_ids=[1, 2], response_ids=[11]),
        )

        pairs, metrics = build_preference_pairs([group], AlgorithmConfig(objective="dpo").dpo)

        assert pairs == []
        assert metrics["dpo/groups_skipped_multistep"] == 1

    def test_skips_ties_by_default(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[11]),
        )

        pairs, metrics = build_preference_pairs([group], AlgorithmConfig(objective="dpo").dpo)

        assert pairs == []
        assert metrics["dpo/groups_skipped_tie"] == 1

    def test_skips_small_reward_gap(self):
        group = _make_group(
            _make_trajectory(reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_trajectory(reward=0.95, prompt_ids=[1, 2], response_ids=[11]),
        )

        cfg = AlgorithmConfig(objective="dpo", dpo={"min_reward_gap": 0.1}).dpo
        pairs, metrics = build_preference_pairs([group], cfg)

        assert pairs == []
        assert metrics["dpo/groups_skipped_small_gap"] == 1


class TestAlgorithmConfigDPO:
    def test_rejects_precomputed_advantage_in_dpo(self):
        with pytest.raises(ValueError, match="use_precomputed_advantage"):
            AlgorithmConfig(objective="dpo", use_precomputed_advantage=True)

    def test_from_config_reads_dpo_fields(self):
        cfg = OmegaConf.create(
            {
                "rllm": {
                    "algorithm": {
                        "objective": "dpo",
                        "adv_estimator": "grpo",
                        "norm_adv_by_std_in_grpo": False,
                        "use_rllm": False,
                        "use_precomputed_advantage": False,
                        "loss_fn": None,
                        "lr_schedule": "constant",
                        "warmup_steps_ratio": 0.0,
                        "dpo": {
                            "beta": 0.2,
                            "pairing_strategy": "best_worst",
                            "min_reward_gap": 0.5,
                            "drop_ties": False,
                        },
                    },
                    "stepwise_advantage": {
                        "mode": "broadcast",
                    },
                }
            }
        )

        algorithm_config = AlgorithmConfig.from_config(cfg)

        assert algorithm_config.objective == TrainingObjective.DPO
        assert algorithm_config.dpo.beta == 0.2
        assert algorithm_config.dpo.min_reward_gap == 0.5
        assert algorithm_config.dpo.drop_ties is False
        assert algorithm_config.norm_adv_by_std_in_grpo is False


class _FakeWorkflowEngine:
    def set_training_step(self, *args, **kwargs):  # noqa: ANN002, D401
        """No-op training step hook."""


class _FakeBackend:
    def __init__(self, episodes: list[Episode]):
        self._episodes = episodes
        self.transform_seen_pairs = None
        self.compute_advantages_called = False
        self.update_policy_called = False

    async def generate_episodes(self, batch, agent_workflow_engine, is_validation=False, **kwargs):  # noqa: ANN001, ANN002, ARG002
        return self._episodes

    def transform_to_backend_batch(self, trainer_state, **kwargs):  # noqa: ANN001, D401
        self.transform_seen_pairs = trainer_state.preference_pairs
        return {"pairs": trainer_state.preference_pairs}

    async def process_backend_batch(self, trainer_state, **kwargs):  # noqa: ANN001, D401
        return None

    async def compute_advantages(self, trainer_state, algorithm_config, **kwargs):  # noqa: ANN001, ARG002, D401
        self.compute_advantages_called = True

    async def update_policy(self, trainer_state, **kwargs):  # noqa: ANN001, D401
        self.update_policy_called = True


class TestUnifiedTrainerObjectiveBranch:
    def _make_trainer(self, backend, objective: str) -> UnifiedTrainer:
        trainer = UnifiedTrainer.__new__(UnifiedTrainer)
        trainer.agent_workflow_engine = _FakeWorkflowEngine()
        trainer.backend = backend
        trainer.algorithm_config = AlgorithmConfig(objective=objective)
        trainer.transform_config = TransformConfig()
        trainer.cf_config = CompactFilteringConfig()
        trainer.rs_config = RejectionSamplingConfig()
        trainer.traj_grouping_hook = _default_traj_grouping_hook
        trainer.tokenizer = None
        return trainer

    def test_dpo_branch_builds_pairs_and_skips_advantages(self):
        episodes = [
            _make_episode("task:0", reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_episode("task:1", reward=0.0, prompt_ids=[1, 2], response_ids=[11]),
        ]
        backend = _FakeBackend(episodes)
        trainer = self._make_trainer(backend, TrainingObjective.DPO.value)
        trainer_state = TrainerState()

        asyncio.run(trainer._train_batch_async(batch=None, trainer_state=trainer_state))

        assert trainer_state.has_preference_pairs
        assert len(trainer_state.preference_pairs) == 1
        assert backend.transform_seen_pairs is trainer_state.preference_pairs
        assert backend.compute_advantages_called is False
        assert backend.update_policy_called is True
        assert trainer_state.metrics["dpo/pairs_built"] == 1

    def test_rl_branch_keeps_advantage_path(self):
        episodes = [
            _make_episode("task:0", reward=1.0, prompt_ids=[1, 2], response_ids=[10]),
            _make_episode("task:1", reward=0.0, prompt_ids=[1, 2], response_ids=[11]),
        ]
        backend = _FakeBackend(episodes)
        trainer = self._make_trainer(backend, TrainingObjective.RL.value)
        trainer_state = TrainerState()

        asyncio.run(trainer._train_batch_async(batch=None, trainer_state=trainer_state))

        assert trainer_state.preference_pairs is None
        assert backend.compute_advantages_called is True
        assert backend.update_policy_called is True


class TestUnifiedTrainerDPOValidation:
    def test_rejects_traj_group_adv_estimator_map_for_dpo(self):
        trainer = UnifiedTrainer.__new__(UnifiedTrainer)
        trainer.rllm_config = OmegaConf.create(
            {
                "algorithm": {
                    "objective": "dpo",
                    "use_precomputed_advantage": False,
                    "adv_estimator": "grpo",
                    "use_rllm": False,
                    "loss_fn": None,
                    "lr_schedule": "constant",
                    "warmup_steps_ratio": 0.0,
                    "dpo": {},
                },
                "rollout": {"n": 2},
                "rejection_sample": {
                    "multiplier": 1,
                    "enable": False,
                    "min_partial_solve_tasks": 1,
                    "min_trajs_per_group": 2,
                },
                "compact_filtering": {},
                "stepwise_advantage": {"mode": "broadcast", "norm_adv_by_std_in_grpo": True},
            }
        )
        trainer.traj_group_adv_estimator_map = {"solver": "grpo"}
        trainer.backend = MagicMock()

        with pytest.raises(ValueError, match="traj_group_adv_estimator_map"):
            trainer._validate_and_setup_configs()
