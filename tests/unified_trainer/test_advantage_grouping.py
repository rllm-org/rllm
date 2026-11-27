"""
Group-relative advantage estimators (GRPO, RLOO, ...) normalise rewards within
rows that share ``non_tensor_batch["uid"]``. The n rollouts of one task must
therefore share that key. These tests pin down which id in the transformed
DataProto has that property, and show what goes wrong when the wrong one is
used (every group is a singleton, so the advantage collapses to the raw reward).
"""

from unittest.mock import MagicMock

import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout import ModelOutput
from rllm.trainer.verl.transform import transform_episodes_to_dataproto


def _episode(task_id: str, rollout_idx: int, reward: float) -> Episode:
    prompt_ids = [1, 2, 3]
    completion_ids = [4, 5]
    step = Step(
        prompt_ids=prompt_ids,
        response_ids=completion_ids,
        model_output=ModelOutput(prompt_ids=prompt_ids, completion_ids=completion_ids),
        reward=reward,
    )
    trajectory = Trajectory(steps=[step], reward=reward)
    return Episode(id=f"{task_id}:{rollout_idx}", trajectories=[trajectory], is_correct=reward > 0)


def _batch():
    # Task A: mixed outcomes (the only kind of group that should carry gradient).
    # Task B: all rollouts correct (group advantage must be exactly zero).
    episodes = [_episode("task_a", i, r) for i, r in enumerate([1.0, 0.0, 1.0, 0.0])]
    episodes += [_episode("task_b", i, 1.0) for i in range(4)]
    engine = MagicMock()
    engine.tokenizer.pad_token_id = 0
    engine.processor = None
    return transform_episodes_to_dataproto(episodes, engine, max_prompt_length=4, max_response_length=4)


def test_trajectory_ids_are_shared_across_rollouts_of_a_task():
    batch = _batch()
    traj_ids = batch.non_tensor_batch["trajectory_ids"]
    step_ids = batch.non_tensor_batch["step_ids"]

    assert set(traj_ids[:4]) == {traj_ids[0]}, "rollouts of task_a must share a trajectory id"
    assert set(traj_ids[4:]) == {traj_ids[4]}, "rollouts of task_b must share a trajectory id"
    assert traj_ids[0] != traj_ids[4], "different tasks must not share a trajectory id"
    # step_ids are keyed on Trajectory.uid, a fresh uuid per rollout, so they cannot group anything.
    assert len(set(step_ids)) == len(step_ids)


def test_grpo_advantage_grouped_by_trajectory_id_has_a_baseline():
    batch = _batch()
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=batch.batch["traj_rewards"],
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["trajectory_ids"],
    )
    per_row = adv.sum(dim=-1) / batch.batch["response_mask"].sum(dim=-1)

    # Mixed group: correct rollouts pushed up, incorrect pushed down, mean zero.
    assert per_row[:4].min() < 0 < per_row[:4].max()
    assert torch.isclose(per_row[:4].mean(), torch.tensor(0.0), atol=1e-6)
    # Uniform group: no signal.
    assert torch.allclose(per_row[4:], torch.zeros(4))


def test_grpo_advantage_grouped_by_step_id_degenerates_to_raw_reward():
    batch = _batch()
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=batch.batch["traj_rewards"],
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["step_ids"],
    )
    per_row = adv.sum(dim=-1) / batch.batch["response_mask"].sum(dim=-1)
    rewards = batch.batch["traj_rewards"].sum(dim=-1)

    # Singleton groups: verl falls back to mean 0 / std 1, so advantage == reward and nothing is ever negative.
    assert torch.allclose(per_row, rewards)
    assert per_row.min() >= 0
    assert not np.any(per_row[4:].numpy() == 0)
