"""
Tests for the verl transform pipeline, focusing on rollout log probs propagation.

Verifies that log probs from ModelOutput are correctly carried through
_process_trajectory → AccumulatedData → _batch_tensors_and_build_data_proto → DataProto,
so that downstream importance sampling and bypass mode work.
"""

from unittest.mock import MagicMock

import numpy as np
import torch

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout import ModelOutput
from rllm.trainer.algorithms.config import CompactFilteringConfig, TransformConfig
from rllm.trainer.algorithms.transform import transform_episodes_to_trajectory_groups
from rllm.trainer.verl.transform import transform_episodes_to_dataproto
from rllm.workflows.workflow import TerminationReason


def _make_mock_rollout_engine(pad_token_id: int = 0):
    """Create a mock VerlEngine with a tokenizer."""
    engine = MagicMock()
    engine.tokenizer.pad_token_id = pad_token_id
    engine.processor = None  # No multimodal processor
    return engine


def _make_episode(
    prompt_ids: list[int],
    completion_ids: list[int],
    logprobs: list[float] | None = None,
    reward: float = 1.0,
    episode_id: str = "task_0:0",
) -> Episode:
    """Create a single-step episode with optional logprobs."""
    model_output = ModelOutput(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
    )
    step = Step(
        prompt_ids=prompt_ids,
        response_ids=completion_ids,
        model_output=model_output,
        reward=reward,
    )
    trajectory = Trajectory(steps=[step], reward=reward)
    return Episode(id=episode_id, trajectories=[trajectory], is_correct=reward > 0)


def test_transform_metrics_handle_all_filtered_groups():
    episodes = [Episode(id=f"task:{i}", trajectories=[], termination_reason=TerminationReason.TIMEOUT) for i in range(2)]
    cf_config = CompactFilteringConfig(enable=True, mask_timeout=True)

    groups, metrics = transform_episodes_to_trajectory_groups(
        episodes,
        TransformConfig(),
        cf_config,
    )

    assert groups == []
    assert metrics["groups/num_groups"] == 0
    assert metrics["groups/num_trajs_after_filter"] == 0
    assert metrics["groups/avg_group_size"] == 0.0
    assert metrics["groups/max_group_size"] == 0
    assert metrics["groups/min_group_size"] == 0


class TestRolloutLogProbsPropagation:
    """Tests that rollout log probs flow through the verl transform pipeline."""

    def test_logprobs_included_in_dataproto(self):
        """When steps have logprobs, DataProto should contain rollout_log_probs."""
        episodes = [
            _make_episode(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5, 6],
                logprobs=[-0.5, -0.3, -0.1],
            ),
            _make_episode(
                prompt_ids=[10, 11],
                completion_ids=[12, 13, 14, 15],
                logprobs=[-0.2, -0.4, -0.6, -0.8],
                episode_id="task_1:0",
            ),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        assert "rollout_log_probs" in batch.batch, "rollout_log_probs should be present when logprobs are available"
        rollout_lp = batch.batch["rollout_log_probs"]
        assert rollout_lp.shape[0] == 2, "Batch size should be 2"
        assert rollout_lp.shape[1] == 8, "Should be padded to max_response_length"

        # First episode: 3 completion tokens, right-padded with 0
        # The actual logprob values should be present in the first positions
        assert torch.isclose(rollout_lp[0, 0], torch.tensor(-0.5))
        assert torch.isclose(rollout_lp[0, 1], torch.tensor(-0.3))
        assert torch.isclose(rollout_lp[0, 2], torch.tensor(-0.1))
        assert rollout_lp[0, 3] == 0.0  # padding

        # Second episode: 4 completion tokens
        assert torch.isclose(rollout_lp[1, 0], torch.tensor(-0.2))
        assert torch.isclose(rollout_lp[1, 1], torch.tensor(-0.4))
        assert torch.isclose(rollout_lp[1, 2], torch.tensor(-0.6))
        assert torch.isclose(rollout_lp[1, 3], torch.tensor(-0.8))
        assert rollout_lp[1, 4] == 0.0  # padding

    def test_no_logprobs_no_rollout_log_probs_key(self):
        """When steps have no logprobs, DataProto should NOT contain rollout_log_probs."""
        episodes = [
            _make_episode(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5, 6],
                logprobs=None,
            ),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        assert "rollout_log_probs" not in batch.batch, "rollout_log_probs should be absent when logprobs are None"

    def test_empty_logprobs_no_rollout_log_probs_key(self):
        """When steps have empty logprobs list, DataProto should NOT contain rollout_log_probs."""
        episodes = [
            _make_episode(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5, 6],
                logprobs=[],
            ),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        assert "rollout_log_probs" not in batch.batch, "rollout_log_probs should be absent when logprobs are empty"

    def test_mixed_logprobs_no_rollout_log_probs_key(self):
        """When some steps have logprobs and others don't, rollout_log_probs should be absent (length mismatch guard)."""
        ep_with = _make_episode(
            prompt_ids=[1, 2, 3],
            completion_ids=[4, 5, 6],
            logprobs=[-0.5, -0.3, -0.1],
            episode_id="task_0:0",
        )
        ep_without = _make_episode(
            prompt_ids=[10, 11],
            completion_ids=[12, 13],
            logprobs=None,
            episode_id="task_1:0",
        )
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto([ep_with, ep_without], engine, max_prompt_length=8, max_response_length=8)

        # Length mismatch: 1 logprob tensor but 2 responses → should not include
        assert "rollout_log_probs" not in batch.batch

    def test_multi_step_trajectory_logprobs(self):
        """Cumulative-prefix multi-step trajectories merge into a single row.

        Step 2's prompt [1,2,3,4,5] prefix-extends step 1's full sequence
        [1,2,3,4], so they merge. The resulting row's response is
        [3, 4, 5, 6, 7, 8] (action₀, delta_obs, action₁) with mask
        [1, 1, 0, 1, 1, 1] and logprobs [-0.1, -0.2, 0, -0.3, -0.4, -0.5].
        """
        model_output_1 = ModelOutput(prompt_ids=[1, 2], completion_ids=[3, 4], logprobs=[-0.1, -0.2])
        model_output_2 = ModelOutput(prompt_ids=[1, 2, 3, 4, 5], completion_ids=[6, 7, 8], logprobs=[-0.3, -0.4, -0.5])
        step1 = Step(prompt_ids=[1, 2], response_ids=[3, 4], model_output=model_output_1, reward=0.0)
        step2 = Step(prompt_ids=[1, 2, 3, 4, 5], response_ids=[6, 7, 8], model_output=model_output_2, reward=1.0)
        trajectory = Trajectory(steps=[step1, step2], reward=1.0)
        episode = Episode(id="task_0:0", trajectories=[trajectory], is_correct=True)

        engine = _make_mock_rollout_engine()
        batch = transform_episodes_to_dataproto([episode], engine, max_prompt_length=8, max_response_length=8)

        assert "rollout_log_probs" in batch.batch
        rollout_lp = batch.batch["rollout_log_probs"]
        # Cumulative-prefix merge → 1 row
        assert rollout_lp.shape[0] == 1

        # Merged logprobs: action₀ (real), observation delta (0.0 placeholder),
        # action₁ (real), then right-padded.
        expected_prefix = [-0.1, -0.2, 0.0, -0.3, -0.4, -0.5]
        for i, exp in enumerate(expected_prefix):
            assert torch.isclose(rollout_lp[0, i], torch.tensor(exp)), (i, rollout_lp[0, i].item())

        # Mask follows the same shape: [1, 1, 0, 1, 1, 1]
        response_mask = batch.batch["response_mask"][0]
        assert response_mask[:6].tolist() == [1, 1, 0, 1, 1, 1]

    def test_other_batch_fields_unchanged(self):
        """Adding logprobs should not affect existing batch fields."""
        episodes = [
            _make_episode(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5, 6],
                logprobs=[-0.5, -0.3, -0.1],
            ),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        # All standard fields should still be present
        for key in ["input_ids", "attention_mask", "position_ids", "prompts", "responses", "response_mask", "traj_rewards", "step_rewards"]:
            assert key in batch.batch, f"Standard field '{key}' should be present"


class TestTaskIdsForGrpoGrouping:
    """Regression tests for #605: GRPO needs the task-level id, not the per-trajectory one.

    `interleave_tasks` gives every `rollout.n` repeat of one task a shared id and encodes it as
    the `task_id:rollout_idx` prefix of `Episode.id` (see `Episode.task_id`). The transform must
    carry that shared id through as its own field so the trainer can group repeats of the same
    task for GRPO's baseline, instead of falling back to `step_ids` (== `Trajectory.uid`, a fresh
    id per trajectory instance that never repeats even for two rollouts of the same task).
    """

    def test_task_ids_shared_across_rollout_repeats(self):
        episodes = [
            _make_episode(prompt_ids=[1, 2], completion_ids=[3, 4], reward=1.0, episode_id="task_0:0"),
            _make_episode(prompt_ids=[1, 2], completion_ids=[3, 4], reward=0.0, episode_id="task_0:1"),
            _make_episode(prompt_ids=[5, 6], completion_ids=[7, 8], reward=1.0, episode_id="task_1:0"),
            _make_episode(prompt_ids=[5, 6], completion_ids=[7, 8], reward=0.0, episode_id="task_1:1"),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        assert "task_ids" in batch.non_tensor_batch
        task_ids = list(batch.non_tensor_batch["task_ids"])
        step_ids = list(batch.non_tensor_batch["step_ids"])

        # The two rollouts of task_0 (rows 0, 1) share one task id, and likewise for task_1
        # (rows 2, 3) -- that is what makes GRPO grouping possible.
        assert task_ids[0] == task_ids[1] == "task_0"
        assert task_ids[2] == task_ids[3] == "task_1"
        assert task_ids[0] != task_ids[2]

        # step_ids is Trajectory.uid: a fresh id per row, so it can never group repeats. If this
        # assertion ever fails, step_ids stopped being trajectory-unique and the bug this test
        # guards against may have changed shape.
        assert len(set(step_ids)) == 4

    def test_grpo_advantage_degenerates_on_step_ids_task_ids_fixes_it(self):
        """Differential against verl's real compute_grpo_outcome_advantage (verl==0.8.0).

        Two rollouts of the same task score 1.0 and 0.0. Grouped correctly, GRPO subtracts the
        pair's own mean (0.5) so the two rollouts get opposite-signed advantages. Grouped by
        step_ids (the pre-fix code), each rollout is alone in its group of one, and
        compute_grpo_outcome_advantage's own size-1 branch hardcodes mean=0/std=1 -- so the
        "advantage" collapses to the raw, unbaselined reward instead.
        """
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        token_level_rewards = torch.tensor(
            [
                [0.0, 1.0],  # task_0 rollout 0, reward 1.0
                [0.0, 0.0],  # task_0 rollout 1, reward 0.0
            ]
        )
        response_mask = torch.ones(2, 2)

        step_ids = np.array(["traj-aaa", "traj-bbb"])  # Trajectory.uid: unique per rollout
        task_ids = np.array(["task_0", "task_0"])  # shared across the pair, as the fix produces

        buggy_advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=step_ids,
        )
        fixed_advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=task_ids,
        )

        # Pre-fix: each rollout is its own group of one -> mean=0, std=1 -> advantage == raw reward.
        assert torch.allclose(buggy_advantages[:, 0], torch.tensor([1.0, 0.0]))

        # Fixed: grouped by task -> mean=0.5, sample std=0.7071 (verl's default
        # norm_adv_by_std_in_grpo=True divides by std) -> baselined, opposite-signed advantages,
        # instead of collapsing to the raw, unbaselined reward.
        assert torch.allclose(fixed_advantages[:, 0], torch.tensor([0.70710678, -0.70710678]))
