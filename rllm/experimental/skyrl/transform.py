"""
Transform functions for converting between rLLM and SkyRL formats.

Follows the same pattern as verl/transform.py.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizer

from rllm.agents.agent import TrajectoryGroup
from rllm.engine.rollout import ModelOutput

if False:  # TYPE_CHECKING:
    from skyrl_train.training_batch import TrainingInputBatch


def transform_trajectory_groups_to_training_input(
    trajectory_groups: list[TrajectoryGroup],
    rollout_engine,  # SkyRLEngine or similar
    max_prompt_length: int,
    max_response_length: int,
) -> TrainingInputBatch:
    """
    Transforms a list of trajectory groups (from running a rLLM workflow) into a SkyRL-compatible TrainingInputBatch.

    Args:
        trajectory_groups: List of trajectory groups to transform.
        rollout_engine: Rollout engine that contains the tokenizer.
        max_prompt_length: The maximum length of the prompts.
        max_response_length: The maximum length of the responses.
    Returns:
        TrainingInputBatch: The TrainingInputBatch built from the trajectory groups.
    """
    from skyrl_train.training_batch import TrainingInputBatch
    from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors

    tokenizer = rollout_engine.tokenizer
    assert tokenizer is not None and hasattr(tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"
    pad_token_id = tokenizer.pad_token_id

    # Extract data from trajectory groups
    prompt_token_ids: list[list[int]] = []
    response_ids: list[list[int]] = []
    rewards: list[list[float]] = []
    loss_masks: list[list[int]] = []
    rollout_logprobs: list[list[float]] | None = []
    uids: list[str] = []  # Collect uids for each trajectory

    for trajectory_group in trajectory_groups:
        task_id = trajectory_group.group_id.split(":")[0]
        
        for trajectory in trajectory_group.trajectories:
            if len(trajectory.steps) == 0:
                continue
            
            # Extract prompt tokens from the first step
            first_step = trajectory.steps[0]
            if not isinstance(first_step.model_output, ModelOutput):
                raise TypeError(f"Step must have a valid model output, but got {type(first_step.model_output)}")
            
            prompt_tokens = first_step.model_output.prompt_ids
            prompt_token_ids.append(prompt_tokens)
            
            # Store uid for this trajectory (used by SkyRL's compute_advantages_and_returns)
            uids.append(task_id)

            # Concatenate all response tokens from all steps
            response_tokens_list: list[int] = []
            response_logprobs_list: list[float] = []
            loss_mask_list: list[int] = []
            reward_list: list[float] = []

            for step in trajectory.steps:
                if not isinstance(step.model_output, ModelOutput):
                    raise TypeError(f"Step must have a valid model output, but got {type(step.model_output)}")
                
                step_response = step.model_output.completion_ids
                response_tokens_list.extend(step_response)

                # Loss mask: 1 for all response tokens
                loss_mask_list.extend([1] * len(step_response))

                # Logprobs if available
                if hasattr(step, "logprobs") and step.logprobs is not None:
                    step_logprobs = step.logprobs
                    response_logprobs_list.extend(step_logprobs)
                else:
                    response_logprobs_list.extend([0.0] * len(step_response))

                # Reward: distribute trajectory reward across steps
                # Put all reward on the last token of the last step
                if step == trajectory.steps[-1]:
                    # Last step gets the reward
                    step_reward = [0.0] * (len(step_response) - 1) + [trajectory.reward if trajectory.reward is not None else 0.0]
                else:
                    step_reward = [0.0] * len(step_response)
                reward_list.extend(step_reward)

            response_ids.append(response_tokens_list)
            rewards.append(reward_list)
            loss_masks.append(loss_mask_list)
            if rollout_logprobs is not None:
                rollout_logprobs.append(response_logprobs_list)

    # Convert to tensors using SkyRL's utility function
    (
        sequences_tensor,
        attention_masks_tensor,
        response_masks_tensor,
        rewards_tensor,
        loss_masks_tensor,
        rollout_logprobs_tensor,
    ) = convert_prompts_responses_to_batch_tensors(
        tokenizer,
        prompt_token_ids,
        response_ids,
        rewards,
        loss_masks,
        rollout_logprobs if rollout_logprobs else None,
    )

    # Create TrainingInputBatch
    training_input = TrainingInputBatch(
        {
            "sequences": sequences_tensor,  # Full trajectories (padded and concatenated prompts and responses)
            "attention_mask": attention_masks_tensor,
            "response_mask": response_masks_tensor,
            "rewards": rewards_tensor,
            "loss_mask": loss_masks_tensor,
            "rollout_logprobs": rollout_logprobs_tensor,
        },
    )

    # Add metadata
    # SkyRL's compute_advantages_and_returns expects uids in metadata
    training_input.metadata = {
        "response_length": response_masks_tensor.shape[1],
        "avg_response_length": sum(len(r) for r in response_ids) / len(response_ids) if response_ids else 0,
        "uids": uids,  # Required by SkyRL's compute_advantages_and_returns
    }

    return training_input

