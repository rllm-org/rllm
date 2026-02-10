"""
Transform functions for converting between rLLM and SkyRL formats.

Follows the same pattern as verl/transform.py.
"""

from __future__ import annotations

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
    from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
    from skyrl_train.training_batch import TrainingInputBatch

    tokenizer = rollout_engine.tokenizer
    assert tokenizer is not None and hasattr(tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"

    # Match the VERL pattern:
    # - "broadcast": use trajectory reward on the final token of the final step
    # - "per_step": use each step reward on the final token of each step
    stepwise_mode = "broadcast"
    config = getattr(rollout_engine, "config", None)
    if config is not None and hasattr(config, "rllm") and hasattr(config.rllm, "stepwise_advantage"):
        configured_mode = config.rllm.stepwise_advantage.get("mode", "broadcast")
        if configured_mode is not None:
            stepwise_mode = str(configured_mode)
    if stepwise_mode not in {"broadcast", "per_step"}:
        raise ValueError(f"Unsupported stepwise mode for SkyRL transform: {stepwise_mode}")

    # Extract data from trajectory groups
    prompt_token_ids: list[list[int]] = []
    response_ids: list[list[int]] = []
    rewards: list[list[float]] = []
    loss_masks: list[list[int]] = []
    rollout_logprobs: list[list[float]] | None = []
    uids: list[str] = []  # Collect uids for each trajectory

    for trajectory_group in trajectory_groups:
        # group_id has the format "task_id:trajectory_name" (e.g. "abc123:solver",
        # "abc123:judge").  SkyRL's GRPO advantage computation groups rows by uid,
        # so we must keep different trajectory names in *separate* groups.
        #
        # Using the full group_id preserves the separation established by
        # _build_trajectory_groups in common/transform.py, matching verl's
        # behaviour where uid = "{task_id}_{trajectory_name}".
        grpo_uid = trajectory_group.group_id  # e.g. "task_id:solver"

        for trajectory in trajectory_group.trajectories:
            if len(trajectory.steps) == 0:
                continue

            # Extract prompt tokens from the first step
            first_step = trajectory.steps[0]
            if not isinstance(first_step.model_output, ModelOutput):
                raise TypeError(f"Step must have a valid model output, but got {type(first_step.model_output)}")

            prompt_tokens = first_step.model_output.prompt_ids
            prompt_token_ids.append(prompt_tokens)

            # Store uid for this trajectory (used by SkyRL's compute_advantages_and_returns).
            # Must be group_id (not bare task_id) so that trajectories with different
            # roles (e.g. solver vs judge) are normalised independently in GRPO.
            uids.append(grpo_uid)

            # Prefer trajectory-level reward; fall back to the final step reward when needed.
            # Some workflows set rewards on steps but leave trajectory.reward unset.
            trajectory_reward = trajectory.reward
            if trajectory_reward is None:
                trajectory_reward = trajectory.steps[-1].reward
            if trajectory_reward is None:
                trajectory_reward = 0.0

            # Concatenate all response tokens from all steps
            response_tokens_list: list[int] = []
            response_logprobs_list: list[float] = []
            loss_mask_list: list[int] = []
            reward_list: list[float] = []

            for step_idx, step in enumerate(trajectory.steps):
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

                if len(step_response) == 0:
                    # No generated tokens in this step; no token-level reward can be assigned.
                    step_reward = []
                else:
                    # Match VERL reward source policy:
                    # - per_step: use each step.reward
                    # - broadcast: use trajectory-level reward on final step only
                    if stepwise_mode == "per_step":
                        token_reward = float(step.reward if step.reward is not None else 0.0)
                    else:  # broadcast
                        is_last_step = step_idx == len(trajectory.steps) - 1
                        token_reward = float(trajectory_reward) if is_last_step else 0.0
                    step_reward = [0.0] * (len(step_response) - 1) + [token_reward]
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
