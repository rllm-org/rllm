"""CPU regressions for workflow GRPO comparison groups (issue #605)."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

from rllm.engine.rollout import ModelOutput
from rllm.trainer.verl.advantage import compute_workflow_grpo_advantage
from rllm.trainer.verl.transform import transform_episodes_to_dataproto
from rllm.types import Episode, Step, Trajectory


def _episode(task, rollout, reward, *, role="solver", segments=1):
    return Episode(
        id=f"{task}:{rollout}",
        trajectories=[
            Trajectory(
                name=role,
                reward=reward,
                steps=[Step(model_output=ModelOutput(prompt_ids=[10 + i], completion_ids=[20, 21])) for i in range(segments)],
            )
        ],
    )


def _batch(episodes):
    engine = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0), processor=None)
    batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)
    batch.batch["token_level_scores"] = batch.batch["traj_rewards"].clone()
    batch.batch["token_level_rewards"] = batch.batch["traj_rewards"].clone()
    return batch


@pytest.mark.parametrize("tasks", [("task",), ("suite:case:a", "suite:case:b")])
def test_rollouts_share_comparison_group_but_keep_unique_identity(tasks):
    episodes = [_episode(task, i, reward) for task in tasks for i, reward in enumerate([1.0, 0.0, 1.0, 0.0])]
    batch = _batch(episodes)

    assert len(set(batch.non_tensor_batch["step_ids"])) == len(episodes)
    assert sorted(Counter(batch.non_tensor_batch["advantage_group_ids"]).values()) == [4] * len(tasks)
    advantages, _ = compute_grpo_outcome_advantage(batch.batch["traj_rewards"], batch.batch["response_mask"], batch.non_tensor_batch["advantage_group_ids"], norm_adv_by_std_in_grpo=True)
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0] * len(tasks)) * (0.5 / (torch.tensor([1.0, 0.0, 1.0, 0.0]).std() + 1e-6))
    torch.testing.assert_close(advantages[:, 0], expected)
    assert torch.all(advantages[:, 2:] == 0)


@pytest.mark.parametrize("normalize", [False, True])
def test_groups_isolate_tasks_and_roles_without_separator_collisions(normalize):
    # Underscore-concatenated keys collide for the first two task/role pairs.
    pairs = [("a_b", "c"), ("a", "b_c"), ("suite:case", "solver"), ("suite:case", "judge")]
    episodes = [_episode(task, rollout, 100.0 * pair_idx + reward, role=role) for pair_idx, (task, role) in enumerate(pairs) for rollout, reward in enumerate([1.0, 0.0, 1.0, 0.0])]
    batch = compute_workflow_grpo_advantage(_batch(episodes), norm_adv_by_std_in_grpo=normalize)
    assert sorted(Counter(batch.non_tensor_batch["uid"]).values()) == [4] * len(pairs)
    scale = 0.5 / (torch.tensor([1.0, 0.0, 1.0, 0.0]).std() + 1e-6) if normalize else 0.5
    torch.testing.assert_close(batch.batch["advantages"][:, 0], torch.tensor([1.0, -1.0, 1.0, -1.0] * len(pairs)) * scale)


@pytest.mark.parametrize("normalize_by_steps", [False, True])
def test_context_resets_compare_each_rollout_once_and_broadcast(normalize_by_steps):
    episodes = [_episode("task", i, reward, segments=count) for i, (reward, count) in enumerate([(1.0, 3), (0.0, 1), (1.0, 2), (0.0, 1)])]
    original = _batch(episodes)
    batch = compute_workflow_grpo_advantage(original, norm_adv_by_std_in_grpo=False, normalize_by_steps=normalize_by_steps)
    expected = torch.tensor([0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5])
    if normalize_by_steps:
        expected /= torch.tensor([3, 3, 3, 1, 2, 2, 1])
    torch.testing.assert_close(batch.batch["advantages"][:, 0], expected)
    torch.testing.assert_close(batch.batch["advantages"][:, 1], expected)
    assert torch.all(batch.batch["advantages"][:, 2:] == 0)
    torch.testing.assert_close(batch.batch["returns"], batch.batch["advantages"])
    np.testing.assert_array_equal(batch.non_tensor_batch["step_ids"], original.non_tensor_batch["step_ids"])


def test_cumulative_turns_mask_observations_and_normalize_by_logical_steps():
    episodes = [_episode("task", i, reward) for i, reward in enumerate([1.0, 0.0])]
    for ep in episodes:
        ep.trajectories[0].steps.append(Step(model_output=ModelOutput(prompt_ids=[10, 20, 21, 30], completion_ids=[40, 41])))
    original = _batch(episodes)
    assert len(original) == 2  # One merged row per rollout.
    batch = compute_workflow_grpo_advantage(original, norm_adv_by_std_in_grpo=False, normalize_by_steps=True)
    torch.testing.assert_close(batch.batch["advantages"], torch.tensor([[0.25, 0.25, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0], [-0.25, -0.25, 0.0, -0.25, -0.25, 0.0, 0.0, 0.0]]))


def test_reward_shaping_is_summed_across_segments_without_repeating_outcome():
    original = _batch([_episode("task", 0, 1.0, segments=2), _episode("task", 1, 0.0)])
    # Rollout scores become 1 - 0.1 - 0.2 = 0.7 and 0 - 0.1 = -0.1.
    original.batch["token_level_rewards"][:, 0] -= torch.tensor([0.1, 0.2, 0.1])
    batch = compute_workflow_grpo_advantage(original, norm_adv_by_std_in_grpo=False)
    torch.testing.assert_close(batch.batch["advantages"][:, 0], torch.tensor([0.4, 0.4, -0.4]))


@pytest.mark.parametrize("shuffle", [False, True])
def test_padding_and_invalid_rows_do_not_change_statistics(shuffle):
    episodes = [_episode("task", i, reward, segments=2 if i == 0 else 1) for i, reward in enumerate([1.0, 0.0, 1.0, 0.0, 1000.0])]
    batch = _batch(episodes)
    batch.non_tensor_batch["is_valid"][-1] = False
    n_valid_rows = len(batch) - 1
    original_size = len(batch)
    batch, pad_size = pad_dataproto_to_divisor(batch, 8)
    assert pad_size > 0
    batch.non_tensor_batch["is_pad_step"][original_size:] = True
    # Deliberately poison padding so accidental inclusion affects the answer.
    batch.batch["token_level_rewards"][original_size:] = 1000.0
    if shuffle:
        batch.reorder(torch.tensor([7, 3, 1, 5, 0, 6, 4, 2]))
    batch = compute_workflow_grpo_advantage(batch, norm_adv_by_std_in_grpo=False)
    assert len(batch) == n_valid_rows
    expected = torch.tensor([0.5 if ep.endswith(":0") or ep.endswith(":2") else -0.5 for ep in batch.non_tensor_batch["episode_ids"]])
    torch.testing.assert_close(batch.batch["advantages"][:, 0], expected)


@pytest.mark.parametrize("rewards", [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [2.0]])
def test_constant_rewards_and_singleton_follow_verl_semantics(rewards):
    original = _batch([_episode("task", i, reward) for i, reward in enumerate(rewards)])
    expected, _ = compute_grpo_outcome_advantage(original.batch["traj_rewards"], original.batch["response_mask"], np.array(["task"] * len(rewards)))
    batch = compute_workflow_grpo_advantage(original)
    torch.testing.assert_close(batch.batch["advantages"], expected)
    assert torch.all(torch.isfinite(batch.batch["advantages"]))


def test_all_filtered_rows_produce_empty_advantages():
    original = _batch([_episode("task", 0, 1.0)])
    original.non_tensor_batch["is_valid"][:] = False
    batch = compute_workflow_grpo_advantage(original)
    assert len(batch) == 0
    assert batch.batch["advantages"].shape == (0, 8)
    assert batch.batch["returns"].shape == (0, 8)
