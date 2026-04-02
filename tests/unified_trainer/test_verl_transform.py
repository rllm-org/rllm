"""
Tests for the verl transform pipeline, focusing on rollout log probs propagation.

Verifies that log probs from ModelOutput are correctly carried through
_process_trajectory → AccumulatedData → _batch_tensors_and_build_data_proto → DataProto,
so that downstream importance sampling and bypass mode work.
"""

import sys
import types
from unittest.mock import MagicMock

import torch

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.experimental.rollout import ModelOutput


def _install_minimal_verl_stubs() -> None:
    if "verl.protocol" in sys.modules and "verl.utils.torch_functional" in sys.modules:
        return

    verl_module = sys.modules.setdefault("verl", types.ModuleType("verl"))
    protocol_module = types.ModuleType("verl.protocol")
    utils_module = types.ModuleType("verl.utils")
    torch_functional_module = types.ModuleType("verl.utils.torch_functional")

    class DataProto:
        def __init__(self, *, tensors, non_tensors, meta_info):
            self.batch = tensors
            self.non_tensor_batch = non_tensors
            self.meta_info = meta_info

        @classmethod
        def from_dict(cls, tensors, non_tensors, meta_info):
            return cls(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def pad_sequence_to_length(batch: torch.Tensor, max_length: int, pad_token_id: int, left_pad: bool = True) -> torch.Tensor:
        current_length = batch.shape[1]
        if current_length >= max_length:
            return batch[:, -max_length:] if left_pad else batch[:, :max_length]

        pad_width = max_length - current_length
        padding = torch.full((batch.shape[0], pad_width), pad_token_id, dtype=batch.dtype, device=batch.device)
        return torch.cat([padding, batch], dim=1) if left_pad else torch.cat([batch, padding], dim=1)

    protocol_module.DataProto = DataProto
    torch_functional_module.pad_sequence_to_length = pad_sequence_to_length
    utils_module.torch_functional = torch_functional_module
    verl_module.protocol = protocol_module
    verl_module.utils = utils_module

    sys.modules["verl.protocol"] = protocol_module
    sys.modules["verl.utils"] = utils_module
    sys.modules["verl.utils.torch_functional"] = torch_functional_module


try:
    from rllm.experimental.verl.transform import transform_episodes_to_dataproto
except ModuleNotFoundError as exc:
    if exc.name != "verl":
        raise
    _install_minimal_verl_stubs()
    from rllm.experimental.verl.transform import transform_episodes_to_dataproto


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
        """Multi-step trajectories should have logprobs for each step row."""
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
        # 2 steps = 2 rows in the batch
        assert rollout_lp.shape[0] == 2

        # Step 1: logprobs [-0.1, -0.2], right-padded
        assert torch.isclose(rollout_lp[0, 0], torch.tensor(-0.1))
        assert torch.isclose(rollout_lp[0, 1], torch.tensor(-0.2))

        # Step 2: logprobs [-0.3, -0.4, -0.5], right-padded
        assert torch.isclose(rollout_lp[1, 0], torch.tensor(-0.3))
        assert torch.isclose(rollout_lp[1, 1], torch.tensor(-0.4))
        assert torch.isclose(rollout_lp[1, 2], torch.tensor(-0.5))

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

    def test_step_response_mask_is_preserved(self):
        """Explicit step-level response masks should flow into Verl response_mask."""
        model_output = ModelOutput(
            prompt_ids=[1, 2],
            completion_ids=[3, 4, 5],
            logprobs=[-0.1, -0.2, -0.3],
        )
        step = Step(
            prompt_ids=[1, 2],
            response_ids=[3, 4, 5],
            response_mask=[1.0, 0.0, 1.0],
            model_output=model_output,
            reward=1.0,
        )
        trajectory = Trajectory(steps=[step], reward=1.0)
        episode = Episode(id="task_0:0", trajectories=[trajectory], is_correct=True)

        engine = _make_mock_rollout_engine()
        batch = transform_episodes_to_dataproto([episode], engine, max_prompt_length=8, max_response_length=8)

        response_mask = batch.batch["response_mask"]
        assert torch.equal(response_mask[0, :3], torch.tensor([1.0, 0.0, 1.0]))
        assert torch.equal(response_mask[0, 3:], torch.zeros(5))
