import uuid

import numpy as np
import torch
from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length

from rllm.agents.agent import Episode, Trajectory, TrajectoryGroup
from rllm.engine.rollout import ModelOutput, VerlEngine
from rllm.experimental.verl.dataclass import AccumulatedData, ProcessedStepData
from rllm.workflows.workflow import TerminationReason


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


def _build_per_step_advantages(response_mask: torch.Tensor, advantages: list[float | list[float]]) -> torch.Tensor:
    """Builds the per-step advantages for a batch of prompts/responses."""
    assert response_mask.shape[0] == len(advantages), "response_mask and advantages must have the same length"
    if isinstance(advantages[0], list):  # TODO(listar2000): support list-based advantages
        raise NotImplementedError("Verl per-step advantages are not supported for list-based advantages yet")
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    return advantages_tensor.unsqueeze(-1) * response_mask


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


def _batch_tensors_and_build_data_proto(accumulated: AccumulatedData, pad_token_id: int, max_prompt_length: int, max_response_length: int, processor=None) -> "DataProto":
    """Batches the tensors from an AccumulatedData.

    Args:
        accumulated: AccumulatedData to batch the tensors from.
        pad_token_id: The token ID to use for padding.
        max_prompt_length: The maximum length to pad the prompts to.
        max_response_length: The maximum length to pad the responses to.
        stepwise_advantage_mode: The mode of stepwise advantage computation.
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

    non_tensors = {
        "episode_ids": np.array(accumulated.episode_ids),  # unique identifier for each rollout
        "trajectory_ids": np.array(accumulated.trajectory_ids),
        "step_ids": np.array(accumulated.step_ids),
        "batch_ids": np.array([str(uuid.uuid4())] * len(accumulated.trajectory_ids)),
        "step_nums": np.array(accumulated.step_nums),
        "is_correct": np.array(accumulated.is_correct),
        "termination_reasons": np.array([x.value for x in accumulated.termination_reasons]),
        "metrics": np.array(accumulated.metrics, dtype=object),
        "is_valid": np.ones(len(accumulated.trajectory_ids), dtype=bool),
        "is_last_step": np.array(accumulated.is_last_step),
        # The padding is done after the transform (in `_pad_dataproto_to_world_size`), so we simply set all to False here
        "is_pad_step": np.zeros(len(accumulated.trajectory_ids), dtype=bool),
    }

    # Include multi_modal_inputs in non_tensors if any are present
    if any(mm_inputs for mm_inputs in accumulated.multi_modal_inputs):
        non_tensors["multi_modal_inputs"] = np.array(accumulated.multi_modal_inputs, dtype=object)

    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": prompts_batch,
        "responses": responses_batch,
        "response_mask": traj_mask,
        "traj_rewards": traj_rewards_batch,
        "step_rewards": step_rewards_batch,
    }

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info={
            "repeat_counts": accumulated.repeat_counts,
        },
    )


def _process_trajectory(trajectory: Trajectory, task_id: str, accumulated: AccumulatedData) -> int:
    """Processes a trajectory and returns an AccumulatedData.

    Args:
        trajectory: Trajectory to process.
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

    # This corresponds to case when we have `per_step` mode for stepwise advantage computation
    traj_reward = 0.0 if trajectory.reward is None else trajectory.reward

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
            advantage=step.advantage,
        )

        accumulated.add_step(
            step_data=step_data,
            trajectory_id=trajectory_id,
            traj_reward=traj_reward,
            step_num=n_steps,
            is_last=step_idx == n_steps - 1,
        )

    return n_steps


def _process_episode(episode: Episode, task_id: str, accumulated: AccumulatedData) -> int:
    """Processes an episode and returns an AccumulatedData.

    Args:
        episode: Episode to process.
        task_id: Task identifier corresponding to the episode.
        accumulated: AccumulatedData to process the episode into.
    Returns:
        repeated_count: The total steps in this episode.
    """
    total_steps = 0

    if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
        # termination hits before an agent finishes it's first step
        # (e.g., the initial prompt exceeds max_prompt_length or a timeout occurs)
        # we delete the episode from the batch by setting repeat_counts to 0
        print(f"Episode {episode.id} has no valid trajectories, dropping it from the batch")
        return 0

    for trajectory in episode.trajectories:
        n_steps = _process_trajectory(trajectory, task_id, accumulated)
        total_steps += n_steps

    # Extend episode-level data for all steps in this episode
    accumulated.episode_ids.extend([episode.id] * total_steps)
    accumulated.is_correct.extend([episode.is_correct] * total_steps)
    termination_reason = episode.termination_reason if episode.termination_reason is not None else TerminationReason.UNKNOWN
    accumulated.termination_reasons.extend([termination_reason] * total_steps)
    accumulated.metrics.extend([episode.metrics] * total_steps)

    return total_steps


def _process_trajectory_group(trajectory_group: TrajectoryGroup, task_id: str, accumulated: AccumulatedData) -> int:
    """Processes a trajectory group and returns an AccumulatedData."""
    total_steps = 0
    for trajectory in trajectory_group.trajectories:
        n_steps = _process_trajectory(trajectory, task_id, accumulated)
        total_steps += n_steps

    # Extend episode-level data for all steps in this trajectory group
    # TrajectoryGroup doesn't have episode-level metadata, so we use reasonable defaults
    # TODO(listar2000): check whether and how we should supplement these info from trajectory groups.
    group_id = trajectory_group.group_id if trajectory_group.group_id else task_id
    accumulated.episode_ids.extend([group_id] * total_steps)
    accumulated.is_correct.extend([False] * total_steps)  # default to False for trajectory groups
    accumulated.termination_reasons.extend([TerminationReason.UNKNOWN] * total_steps)
    accumulated.metrics.extend([{}] * total_steps)  # empty metrics for trajectory groups

    return total_steps


def transform_episodes_to_dataproto(
    episodes: list[Episode],
    rollout_engine: VerlEngine,
    max_prompt_length: int,
    max_response_length: int,
) -> DataProto:
    """
    Transforms a list of episodes (from running a rLLM workflow) into a verl-compatible DataProto.

    Args:
        episodes: List of episodes to transform.
        rollout_engine: Rollout engine that contains the tokenizer and (optional) multimodal processor.
        max_prompt_length: The maximum length of the prompts.
        max_response_length: The maximum length of the responses.
        stepwise_advantage_mode: The mode of stepwise advantage computation.
    Returns:
        DataProto: The DataProto built from the episodes.
    """
    tokenizer = rollout_engine.tokenizer
    processor = getattr(rollout_engine, "processor", None)

    accumulated = AccumulatedData()
    for episode in episodes:
        task_id = episode.id.split(":")[0]
        total_steps = _process_episode(episode, task_id, accumulated)
        accumulated.repeat_counts.append(total_steps)

    assert hasattr(tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"
    pad_token_id = tokenizer.pad_token_id
    return _batch_tensors_and_build_data_proto(accumulated, pad_token_id, max_prompt_length, max_response_length, processor)


# TODO: extract common logic from transform_episodes_to_dataproto and transform_trajectory_groups_to_dataproto
def transform_trajectory_groups_to_dataproto(
    trajectory_groups: list[TrajectoryGroup],
    rollout_engine: VerlEngine,
    max_prompt_length: int,
    max_response_length: int,
) -> DataProto:
    """
    Transforms a list of trajectory groups (from running a rLLM workflow) into a verl-compatible DataProto.
    """
    tokenizer = rollout_engine.tokenizer
    processor = getattr(rollout_engine, "processor", None)

    accumulated = AccumulatedData()
    for trajectory_group in trajectory_groups:
        task_id = trajectory_group.group_id.split(":")[0]
        total_steps = _process_trajectory_group(trajectory_group, task_id, accumulated)
        accumulated.repeat_counts.append(total_steps)

    assert tokenizer is not None and hasattr(tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"
    pad_token_id = tokenizer.pad_token_id
    return _batch_tensors_and_build_data_proto(accumulated, pad_token_id, max_prompt_length, max_response_length, processor)


def update_dataproto_with_advantages(batch: DataProto, container: list[Episode | TrajectoryGroup], mode: str = "broadcast") -> DataProto:
    """
    Updates a DataProto with advantages. Useful when we use rLLM-native advantage computation,
    after which we need to update the DataProto with the advantages.
    """
    # flatten the steps in the container exactly like how we build it up
    advantages = []
    for item in container:
        for trajectory in item.trajectories:
            for step in trajectory.steps:
                advantages.append(step.advantage)

    assert len(advantages) == len(batch.non_tensor_batch["trajectory_ids"]), "Advantages must have the same length as the number of total steps"
    advantage_tensor = _build_per_step_advantages(batch.batch["response_mask"], advantages)
    batch.batch["advantages"] = advantage_tensor
    batch.batch["returns"] = advantage_tensor

    # TODO(listar2000): we should support `token_level_scores` from the `Step` attribute level.
    # we also need to implement the `kl_penalty` logic used in `verl`.
    if mode == "per_step":
        batch.batch["token_level_scores"] = batch.batch["step_rewards"]
        batch.batch["token_level_rewards"] = batch.batch["step_rewards"]
    elif mode == "broadcast":
        batch.batch["token_level_scores"] = batch.batch["traj_rewards"]
        batch.batch["token_level_rewards"] = batch.batch["traj_rewards"]
    else:
        raise ValueError(f"Stepwise advantage mode {mode} not supported")

    return batch
