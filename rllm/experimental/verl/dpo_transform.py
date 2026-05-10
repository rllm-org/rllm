"""Verl DataProto transform for strict DPO preference pairs."""

from __future__ import annotations

import uuid

import numpy as np
import torch
from verl.protocol import DataProto

from rllm.experimental.common.preference import PreferencePair
from rllm.experimental.rollout import VerlEngine
from rllm.experimental.verl.transform import (
    _build_step_and_trajectory_rewards,
    _handle_multimodal_position_ids,
    _pad_sequence_batch,
    _retrieve_batch_attention_masks,
)
from rllm.types import Trajectory
from rllm.workflows.workflow import TerminationReason


def _trajectory_row_tokens(trajectory: Trajectory) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
    if len(trajectory.steps) != 1:
        raise ValueError(f"DPO transform only supports single-step trajectories, got {len(trajectory.steps)} steps for trajectory {trajectory.uid}")

    step = trajectory.steps[0]
    prompt_ids = list(step.prompt_ids)
    response_ids = list(step.response_ids)
    if not prompt_ids:
        raise ValueError(f"DPO trajectory {trajectory.uid} has no prompt_ids")
    if not response_ids:
        raise ValueError(f"DPO trajectory {trajectory.uid} has no response_ids")

    prompt = torch.tensor(prompt_ids, dtype=torch.long)
    response = torch.tensor(response_ids, dtype=torch.long)
    response_mask = torch.ones(len(response_ids), dtype=torch.long)

    rollout_logprobs = None
    if step.logprobs:
        if len(step.logprobs) != len(response_ids):
            raise ValueError(f"DPO trajectory {trajectory.uid} has {len(response_ids)} response tokens but {len(step.logprobs)} rollout logprobs")
        rollout_logprobs = torch.tensor(step.logprobs, dtype=torch.float32)

    multi_modal_inputs = {}
    if step.model_output is not None:
        multi_modal_inputs = step.model_output.multi_modal_inputs or {}

    return prompt, response, response_mask, rollout_logprobs, multi_modal_inputs


def transform_preference_pairs_to_dataproto(
    pairs: list[PreferencePair],
    rollout_engine: VerlEngine,
    max_prompt_length: int,
    max_response_length: int,
) -> DataProto:
    """Transform strict DPO chosen/rejected pairs into a Verl ``DataProto``.

    Rows are emitted in chosen/rejected order and kept adjacent. The DPO backend
    sends this batch to the actor as one unshuffled minibatch so the loss can
    compute pairwise margins without treating DPO as a generic trainer objective.
    """
    if not pairs:
        raise ValueError("Cannot transform an empty DPO preference-pair list")

    tokenizer = rollout_engine.tokenizer
    processor = getattr(rollout_engine, "processor", None)
    assert hasattr(tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"
    pad_token_id = tokenizer.pad_token_id

    prompts: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []
    response_masks: list[torch.Tensor] = []
    rollout_logprobs: list[torch.Tensor | None] = []
    multi_modal_inputs: list[dict] = []
    step_rewards: list[float] = []
    traj_rewards: list[float] = []

    episode_ids: list[str] = []
    trajectory_ids: list[str] = []
    step_ids: list[str] = []
    step_nums: list[int] = []
    is_correct: list[bool] = []
    termination_reasons: list[str] = []
    metrics: list[dict] = []
    group_roles: list[str] = []
    pair_ids: list[str] = []
    is_chosen: list[bool] = []
    reward_gaps: list[float] = []
    pair_indices: list[int] = []

    for pair_idx, pair in enumerate(pairs):
        chosen_prompt = pair.chosen.steps[0].prompt_ids
        rejected_prompt = pair.rejected.steps[0].prompt_ids
        if chosen_prompt != rejected_prompt:
            raise ValueError(f"DPO pair {pair.group_id} has mismatched prompt_ids")

        pair_id = f"{pair.group_id}:{pair_idx}"
        for chosen_flag, trajectory in ((True, pair.chosen), (False, pair.rejected)):
            prompt, response, response_mask, row_logprobs, row_multi_modal_inputs = _trajectory_row_tokens(trajectory)

            prompts.append(prompt)
            responses.append(response)
            response_masks.append(response_mask)
            rollout_logprobs.append(row_logprobs)
            multi_modal_inputs.append(row_multi_modal_inputs)

            reward = 0.0 if trajectory.reward is None else float(trajectory.reward)
            step_rewards.append(reward)
            traj_rewards.append(reward)

            episode_ids.append(pair.task_id)
            trajectory_ids.append(trajectory.uid)
            step_ids.append(trajectory.steps[0].id)
            step_nums.append(1)
            is_correct.append(chosen_flag)
            termination_reasons.append(TerminationReason.UNKNOWN.value)
            metrics.append({})
            group_roles.append(pair.role)
            pair_ids.append(pair_id)
            is_chosen.append(chosen_flag)
            reward_gaps.append(pair.reward_gap)
            pair_indices.append(pair_idx)

    prompts_batch = _pad_sequence_batch(prompts, pad_token_id, max_prompt_length, left_pad=True)
    responses_batch = _pad_sequence_batch(responses, pad_token_id, max_response_length, left_pad=False)
    input_ids = torch.concat([prompts_batch, responses_batch], dim=1)

    prompts_mask = _retrieve_batch_attention_masks(prompts_batch, pad_token_id, max_prompt_length)
    responses_mask = _retrieve_batch_attention_masks(responses_batch, pad_token_id, max_response_length)
    attention_mask = torch.concat([prompts_mask, responses_mask], dim=1)

    if processor is not None and hasattr(processor, "image_processor") and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        position_ids = _handle_multimodal_position_ids(
            processor=processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=multi_modal_inputs,
        )
    else:
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    response_mask_batch = _pad_sequence_batch(response_masks, 0, max_response_length, left_pad=False)
    step_rewards_batch, traj_rewards_batch = _build_step_and_trajectory_rewards(step_rewards, traj_rewards, responses_batch, responses)

    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": prompts_batch,
        "responses": responses_batch,
        "response_mask": response_mask_batch,
        "traj_rewards": traj_rewards_batch,
        "step_rewards": step_rewards_batch,
        "dpo_is_chosen": torch.tensor(is_chosen, dtype=torch.bool),
        "dpo_pair_indices": torch.tensor(pair_indices, dtype=torch.long),
        "dpo_pair_weights": torch.ones(len(pair_indices), dtype=torch.float32),
    }

    if all(row_logprobs is not None for row_logprobs in rollout_logprobs):
        tensors["rollout_log_probs"] = _pad_sequence_batch(
            [row_logprobs for row_logprobs in rollout_logprobs if row_logprobs is not None],
            0,
            max_response_length,
            left_pad=False,
        )

    non_tensors = {
        "episode_ids": np.array(episode_ids, dtype=object),
        "trajectory_ids": np.array(trajectory_ids, dtype=object),
        "step_ids": np.array(step_ids, dtype=object),
        "batch_ids": np.array([str(uuid.uuid4())] * len(trajectory_ids), dtype=object),
        "step_nums": np.array(step_nums),
        "is_correct": np.array(is_correct),
        "termination_reasons": np.array(termination_reasons, dtype=object),
        "metrics": np.array(metrics, dtype=object),
        "is_valid": np.ones(len(trajectory_ids), dtype=bool),
        "is_last_step": np.ones(len(trajectory_ids), dtype=bool),
        "is_pad_step": np.zeros(len(trajectory_ids), dtype=bool),
        "group_roles": np.array(group_roles, dtype=object),
        "pair_ids": np.array(pair_ids, dtype=object),
        "is_chosen": np.array(is_chosen),
        "reward_gap": np.array(reward_gaps, dtype=np.float32),
    }

    if any(row_multi_modal_inputs for row_multi_modal_inputs in multi_modal_inputs):
        non_tensors["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info={
            "dpo_num_pairs": len(pairs),
        },
    )
