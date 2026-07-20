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
from rllm.trainer.verl.transform import _build_token_advantages, transform_episodes_to_dataproto
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
        assert "advantages" not in batch.batch


def test_verl_preserves_token_advantages_through_prefix_merge():
    model_output_1 = ModelOutput(prompt_ids=[1, 2], completion_ids=[3, 4], logprobs=[-0.1, -0.2])
    model_output_2 = ModelOutput(prompt_ids=[1, 2, 3, 4, 5], completion_ids=[6, 7], logprobs=[-0.3, -0.4])
    step1 = Step(model_output=model_output_1, advantage=[0.1, 0.2])
    step2 = Step(model_output=model_output_2, advantage=[0.3, 0.4])
    trajectory = Trajectory(steps=[step1, step2], reward=1.0)
    episode = Episode(id="task:0", trajectories=[trajectory])

    batch = transform_episodes_to_dataproto([episode], _make_mock_rollout_engine(), max_prompt_length=8, max_response_length=8)

    # Merged response is [3, 4, 5, 6, 7]: action, observation, action.
    assert torch.allclose(batch.batch["advantages"][0], torch.tensor([0.1, 0.2, 0.0, 0.3, 0.4, 0.0, 0.0, 0.0]))


def test_build_token_advantages_pads_truncates_and_masks_rows():
    response_mask = torch.tensor(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    advantages = [[0.1, 0.2], [1.0, 2.0, 3.0, 4.0, 5.0], []]

    result = _build_token_advantages(response_mask, advantages)

    assert result.dtype == torch.float32
    assert torch.allclose(
        result,
        torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )

    empty_result = _build_token_advantages(torch.empty((0, 4), dtype=torch.bool), [])
    assert empty_result.shape == (0, 4)


def test_verl_expands_legacy_scalar_advantage_input():
    episode = _make_episode(prompt_ids=[1, 2], completion_ids=[3, 4], logprobs=[-0.1, -0.2])
    episode.trajectories[0].steps[0].advantage = 0.5

    batch = transform_episodes_to_dataproto([episode], _make_mock_rollout_engine(), max_prompt_length=4, max_response_length=4)

    assert torch.allclose(batch.batch["advantages"][0], torch.tensor([0.5, 0.5, 0.0, 0.0]))


def test_verl_token_advantages_move_with_split_rows_and_reordering():
    step1 = Step(model_output=ModelOutput(prompt_ids=[1], completion_ids=[2], logprobs=[-0.1]), advantage=[0.1])
    # This prompt is not a prefix extension, so it becomes a second row.
    step2 = Step(model_output=ModelOutput(prompt_ids=[9], completion_ids=[10, 11], logprobs=[-0.2, -0.3]), advantage=[0.2, 0.3])
    trajectory = Trajectory(steps=[step1, step2], reward=1.0)
    episode = Episode(id="task:0", trajectories=[trajectory])

    batch = transform_episodes_to_dataproto([episode], _make_mock_rollout_engine(), max_prompt_length=4, max_response_length=4)
    batch = batch.select_idxs(np.array([1, 0]))  # mimic backend balancing

    assert torch.allclose(batch.batch["advantages"][0], torch.tensor([0.2, 0.3, 0.0, 0.0]))
    assert torch.allclose(batch.batch["advantages"][1], torch.tensor([0.1, 0.0, 0.0, 0.0]))


def test_verl_token_advantages_robust_to_pad_equal_action_token():
    # pad_token_id may equal a real emitted token (common when pad == eos).
    # Targets are built alongside each raw response row, before padding, so the
    # shorter [5] action cannot collide with the [5, pad] action.
    pad = 2
    step1 = Step(model_output=ModelOutput(prompt_ids=[1], completion_ids=[5], logprobs=[-0.1]), advantage=[0.1])
    step2 = Step(model_output=ModelOutput(prompt_ids=[9], completion_ids=[5, pad], logprobs=[-0.2, -0.3]), advantage=[0.2, 0.3])
    trajectory = Trajectory(steps=[step1, step2], reward=1.0)
    episode = Episode(id="task:0", trajectories=[trajectory])

    engine = _make_mock_rollout_engine(pad_token_id=pad)
    batch = transform_episodes_to_dataproto([episode], engine, max_prompt_length=4, max_response_length=4)

    rows = [r for r in batch.batch["advantages"]]
    assert any(torch.allclose(r, torch.tensor([0.1, 0.0, 0.0, 0.0])) for r in rows), rows
    assert any(torch.allclose(r, torch.tensor([0.2, 0.3, 0.0, 0.0])) for r in rows), rows
