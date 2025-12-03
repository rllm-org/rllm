import uuid

import numpy as np
import torch

from rllm.agents.agent import Episode, Trajectory
from rllm.engine.rollout import ModelOutput
from rllm.trainer.verl.verl_data_processor.dataclass import AccumulatedData, CompactFilteringConfig, ProcessedStepData
from rllm.workflows.workflow import TerminationReason
from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length


def _pad_sequence_batch(sequences: list[torch.Tensor], pad_token_id: int, max_length: int, left_pad: bool = True) -> torch.Tensor:
    """Pads a list of sequences to a maximum length.

    Args:
        sequences: List of sequences to pad.
        pad_token_id: The token ID to use for padding.
        max_length: The maximum length to pad to.
        left_pad: Whether to pad on the left or right.
    Returns:
        torch.Tensor: The padded sequences.
    """
    if left_pad:
        rev_sequences = [torch.flip(seq, dims=[0]) for seq in sequences]
        batch = torch.nn.utils.rnn.pad_sequence(rev_sequences, batch_first=True, padding_value=pad_token_id).flip(dims=[1])
    else:
        batch = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)

    batch = pad_sequence_to_length(batch, max_length, pad_token_id, left_pad=left_pad)
    # additional truncation check
    batch = batch[:, -max_length:] if left_pad else batch[:, :max_length]
    return batch


def _retrieve_batch_attention_masks(batch: torch.Tensor, pad_token_id: int, max_length: int) -> torch.Tensor:
    """Retrieves the attention masks for a batch of prompts/responses.

    Note that in original implementation, this operation is padding-aware, i.e. it has DIFFERENT behavior for left-pad (prompts)
    and right-pad (responses) sequences. This is to ensure compatibility with the `input_ids` constructed with results from function `_pad_sequence_batch`.

    The current version simply uses the constructed `promits_batch` (`responses_batch`) instead of the original `prompts` (`responses`) lengths.
    """
    assert len(batch.shape) == 2, f"batch must be a 2D tensor, but got {batch.shape}"
    assert batch.shape[1] == max_length, f"input batch must have been padded to {max_length}, but got {batch.shape[1]}"
    return batch != pad_token_id


def _build_step_and_trajectory_rewards(step_rewards: list[float], trajectory_rewards: list[float], responses_batch: torch.Tensor, responses: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the step and trajectory rewards for a batch of prompts/responses.

    Args:
        step_rewards: List of step rewards.
        trajectory_rewards: List of trajectory rewards.
        shape: Shape of the step and trajectory rewards. Should be (bs, max_response_length).
        responses: List of responses. Should be a list of tensors with shape (seq_len,).
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The step and trajectory rewards.
    """
    assert len(step_rewards) == len(trajectory_rewards), "step_rewards and trajectory_rewards must have the same length"

    step_rewards_batch = torch.zeros(responses_batch.shape, dtype=torch.float32)  # shape: [bs, max_response_length]
    trajectory_rewards_batch = torch.zeros(responses_batch.shape, dtype=torch.float32)  # shape: [bs, max_response_length]
    for i, (step_reward, trajectory_reward, response) in enumerate(zip(step_rewards, trajectory_rewards, responses, strict=False)):
        resp_len = len(response)
        if resp_len > 0 and resp_len <= responses_batch.shape[1]:
            step_rewards_batch[i, resp_len - 1] = step_reward
            trajectory_rewards_batch[i, resp_len - 1] = trajectory_reward

    return step_rewards_batch, trajectory_rewards_batch


def _compact_filtering(termination_reasons: list[TerminationReason], cf_config: CompactFilteringConfig) -> list[bool]:
    """Performs compact filtering based on the termination reasons and the compact filtering configuration.

    Args:
        termination_reasons: List of termination reasons.
        cf_config: Compact filtering configuration.
    Returns:
        List of booleans indicating whether each episode is valid.
    """
    if not cf_config.enable:
        return [True] * len(termination_reasons)

    return [not cf_config.should_mask(reason) for reason in termination_reasons]


def _handle_multimodal_position_ids(processor, input_ids: torch.Tensor, attention_mask: torch.Tensor, multi_modal_inputs: list[dict]) -> torch.Tensor:
    """Handle multimodal position ids calculation. Borrowed from verl.utils.dataset.rl_dataset.py

    Args:
        processor: The multimodal processor (e.g., Qwen2VLProcessor or Qwen3VLProcessor).
        input_ids: Tensor of input token IDs with shape (batch_size, seq_length).
        attention_mask: Tensor of attention masks with shape (batch_size, seq_length).
        multi_modal_inputs: List of dicts containing multimodal inputs per batch item.
    Returns:
        torch.Tensor: Position IDs tensor with shape (batch_size, 4, seq_length) for Qwen-VL models.
    """
    batch_size = input_ids.shape[0]
    position_ids_list = []

    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        # qwen-vl mrope
        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index

        for i in range(batch_size):
            model_inputs = multi_modal_inputs[i] if i < len(multi_modal_inputs) else {}
            vision_position_ids = get_rope_index(
                processor,
                input_ids=input_ids[i],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[i],
            )  # (3, seq_length)
            valid_mask = attention_mask[i].bool()
            text_position_ids = torch.ones((1, len(input_ids[i])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids_list.append(torch.cat((text_position_ids, vision_position_ids), dim=0))  # (4, seq_length)

    else:
        # Fallback: should not reach here if called correctly
        raise ValueError(f"Unsupported processor type: {processor.__class__.__name__ if processor else None}")

    # Stack all position_ids to form batch: (batch_size, 4, seq_length)
    position_ids = torch.stack(position_ids_list, dim=0)
    return position_ids


def _batch_tensors_and_build_data_proto(accumulated: AccumulatedData, pad_token_id: int, max_prompt_length: int, max_response_length: int, cf_config: CompactFilteringConfig, processor=None) -> "DataProto":
    """Batches the tensors from an AccumulatedData.

    Args:
        accumulated: AccumulatedData to batch the tensors from.
        pad_token_id: The token ID to use for padding.
        max_prompt_length: The maximum length to pad the prompts to.
        max_response_length: The maximum length to pad the responses to.
        cf_config: Compact filtering configuration.
        processor: Optional multimodal processor for handling position IDs (e.g., Qwen2VLProcessor).
    Returns:
        DataProto: The DataProto built from the AccumulatedData.
    """
    prompts_batch = _pad_sequence_batch(accumulated.prompts, pad_token_id, max_prompt_length, left_pad=True)  # shape: [bs, max_prompt_length]
    responses_batch = _pad_sequence_batch(accumulated.responses, pad_token_id, max_response_length, left_pad=False)  # shape: [bs, max_response_length]
    input_ids = torch.concat([prompts_batch, responses_batch], dim=1)  # shape: [bs, max_prompt_length + max_response_length]

    prompts_mask = _retrieve_batch_attention_masks(prompts_batch, pad_token_id, max_prompt_length)
    responses_mask = _retrieve_batch_attention_masks(responses_batch, pad_token_id, max_response_length)
    attention_mask = torch.concat([prompts_mask, responses_mask], dim=1)  # shape: [bs, max_prompt_length + max_response_length]

    # Handle position_ids: use multimodal handler if processor is available
    if processor is not None and hasattr(processor, "image_processor") and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        position_ids = _handle_multimodal_position_ids(
            processor=processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=accumulated.multi_modal_inputs,
        )
    else:
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask  # shape: [bs, max_prompt_length + max_response_length]

    traj_mask = _pad_sequence_batch(accumulated.traj_mask, 0, max_response_length, left_pad=False)  # shape: [bs, max_response_length]

    step_rewards_batch, traj_rewards_batch = _build_step_and_trajectory_rewards(accumulated.step_rewards, accumulated.traj_rewards, responses_batch, accumulated.responses)  # shape: [bs, max_response_length]

    is_valid = _compact_filtering(accumulated.termination_reasons, cf_config)

    non_tensors = {
        "episode_ids": np.array(accumulated.episode_ids),
        "trajectory_ids": np.array(accumulated.trajectory_ids),
        "step_ids": np.array(accumulated.step_ids),
        "batch_ids": np.array([str(uuid.uuid4())] * len(accumulated.episode_ids)),
        "step_nums": np.array(accumulated.step_nums),
        "is_correct": np.array(accumulated.is_correct),
        "termination_reasons": np.array([x.value for x in accumulated.termination_reasons]),
        "metrics": np.array(accumulated.metrics),
        "is_valid": np.array(is_valid),
        "is_last_step": np.array(accumulated.is_last_step),
        # The padding is done after the transform (in `_pad_dataproto_to_world_size`), so we simply set all to False here
        "is_pad_step": np.array([False] * len(accumulated.episode_ids)),
    }

    # Include multi_modal_inputs in non_tensors if any are present
    if any(mm_inputs for mm_inputs in accumulated.multi_modal_inputs):
        non_tensors["multi_modal_inputs"] = np.array(accumulated.multi_modal_inputs, dtype=object)

    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts_batch,
            "responses": responses_batch,
            "response_mask": traj_mask,
            "traj_rewards": traj_rewards_batch,
            "step_rewards": step_rewards_batch,
        },
        non_tensors=non_tensors,
        meta_info={
            "repeat_counts": accumulated.repeat_counts,
        },
    )


def _process_trajectory(trajectory: Trajectory, episode: Episode, task_id: str, accumulated: AccumulatedData) -> int:
    """Processes a trajectory and returns an AccumulatedData.

    Args:
        trajectory: Trajectory to process.
        episode: (Parent) episode corresponding to the trajectory.
        task_id: Task identifier corresponding to the episode.
        accumulated: AccumulatedData to process the trajectory into.
    Returns:
        n_steps: The number of steps in the trajectory.
    """
    name = trajectory.name
    trajectory_id = f"{task_id}_{name}"
    if len(trajectory.steps) == 0:
        print(f"Trajectory {trajectory_id} has no steps, skipping")
        return 0

    n_steps = len(trajectory.steps)

    for step_idx, step in enumerate(trajectory.steps):
        if not isinstance(step.model_output, ModelOutput):
            raise TypeError(f"Step {step_idx} in trajectory {trajectory_id} must have a valid model output, but got {type(step.model_output)}")

        prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
        mask = torch.ones_like(response_ids, dtype=torch.long)
        step_reward = step.reward
        # Extract multimodal inputs if available
        multi_modal_inputs = step.model_output.multi_modal_inputs or {}
        # Construct step_id from trajectory_id and step index
        # Format: "{trajectory_id}_step{step_idx}"
        # Example: "abc123_solver_step0", "abc123_judge_step1"
        # Since trajectory_id doesn't contain rollout info, step_id doesn't either
        step_id = f"{trajectory_id}_step{step_idx}"

        step_data = ProcessedStepData(
            prompt=prompt_ids,
            response=response_ids,
            mask=mask,
            step_reward=step_reward,
            step_id=step_id,
            multi_modal_inputs=multi_modal_inputs,
        )

        accumulated.add_step(
            step_data=step_data,
            episode_id=episode.id,
            trajectory_id=trajectory_id,
            traj_reward=trajectory.reward,
            step_num=n_steps,
            is_last=step_idx == n_steps - 1,
            is_correct=episode.is_correct,
            termination_reason=episode.termination_reason,
            metrics=episode.metrics,
        )

    return n_steps


def _process_episode(episode: Episode, task_id: str, accumulated: AccumulatedData) -> int:
    """Processes an episode and returns an AccumulatedData.

    Args:
        episode: Episode to process.
        task_id: Task identifier corresponding to the episode.
        accumulated: AccumulatedData to process the episode into.
    Returns:
        repeated_count: The number of times the episode is repeated.
    """
    total_steps = 0
    if episode is None:
        print(f"Episode with task_id {task_id} is None (failed task), dropping it from the batch")
        return 0

    if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
        # termination hits before an agent finishes it's first step
        # (e.g., the initial prompt exceeds max_prompt_length or a timeout occurs)
        # we delete the episode from the batch by setting repeat_counts to 0
        print(f"Episode {episode.id} has no valid trajectories, dropping it from the batch")
        return 0

    if episode.termination_reason is None:
        episode.termination_reason = TerminationReason.UNKNOWN

    for trajectory in episode.trajectories:
        n_steps = _process_trajectory(trajectory, episode, task_id, accumulated)
        total_steps += n_steps

    return total_steps


def transform_episodes_for_verl(
    episodes: list[Episode],
    task_ids: np.ndarray,
    tokenizer,
    max_prompt_length: int,
    max_response_length: int,
    cf_config: CompactFilteringConfig,
    processor=None,
) -> DataProto:
    """
    Transforms a list of episodes (from running a rLLM workflow) into a verl-compatible DataProto.

    Args:
        episodes: List of episodes to transform.
        task_ids: Array of task identifiers corresponding to the episodes.
        tokenizer: Tokenizer to use for tokenizing the episodes.
        max_prompt_length: The maximum length of the prompts.
        max_response_length: The maximum length of the responses.
        cf_config: Compact filtering configuration.
        processor: Optional multimodal processor for handling position IDs (e.g., Qwen2VLProcessor).
    Returns:
        DataProto: The DataProto built from the episodes.
    """
    accumulated = AccumulatedData()
    for i, episode in enumerate(episodes):
        total_steps = _process_episode(episode, task_ids[i], accumulated)
        accumulated.repeat_counts.append(total_steps)

    assert hasattr(tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"
    pad_token_id = tokenizer.pad_token_id
    return _batch_tensors_and_build_data_proto(accumulated, pad_token_id, max_prompt_length, max_response_length, cf_config, processor)
