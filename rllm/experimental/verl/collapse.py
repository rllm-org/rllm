import logging
from dataclasses import dataclass

import torch

from rllm.agents.agent import Trajectory
from rllm.experimental.verl.dataclass import ProcessedStepData

logger = logging.getLogger(__name__)


@dataclass
class CollapsedTrajectoryStep:
    step_data: ProcessedStepData
    trajectory_id: str
    traj_reward: float
    step_num: int
    is_last_step: bool
    group_role: str


def is_prefix(prefix_prompt: torch.Tensor, prefix_response: torch.Tensor, step_prompt: torch.Tensor) -> bool:
    """Check if prefix prompt+response is a prefix of the current step prompt."""
    prefix_prompt_len = len(prefix_prompt)
    prefix_response_len = len(prefix_response)
    prefix_len = prefix_prompt_len + prefix_response_len

    if prefix_len > len(step_prompt):
        return False

    return torch.equal(prefix_prompt, step_prompt[:prefix_prompt_len]) and torch.equal(
        prefix_response, step_prompt[prefix_prompt_len:prefix_len]
    )


def collapse_trajectory_steps(trajectory: Trajectory, task_id: str) -> list[CollapsedTrajectoryStep]:
    """Collapse prefix-chained trajectory steps into training rows."""
    name = trajectory.name
    trajectory_id = f"{task_id}_{name}"
    n_steps = len(trajectory.steps)
    traj_reward = 0.0 if trajectory.reward is None else trajectory.reward

    processed_steps: list[CollapsedTrajectoryStep] = []
    last_modified_idx = -1

    for step_idx, step in enumerate(trajectory.steps):
        if step.model_output is None or step.model_output.prompt_ids is None:
            logger.warning(f"Step {step_idx} in trajectory {trajectory_id} has no valid model_output, skipping")
            continue

        prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)

        merged = False
        for i, collapsed_step in enumerate(processed_steps):
            prefix_data = collapsed_step.step_data
            if is_prefix(prefix_data.prompt, prefix_data.response, prompt_ids):
                prefix_len = len(prefix_data.prompt) + len(prefix_data.response)
                gap_tokens = prompt_ids[prefix_len:]

                new_response = torch.cat([prefix_data.response, gap_tokens, response_ids])
                new_mask = torch.cat(
                    [
                        prefix_data.mask,
                        torch.zeros(len(gap_tokens), dtype=torch.long),
                        torch.ones(len(response_ids), dtype=torch.long),
                    ]
                )

                prev_logprobs = prefix_data.logprobs
                current_logprobs = step.model_output.logprobs
                if prev_logprobs is not None and current_logprobs is not None:
                    merged_logprobs = list(prev_logprobs) + [0.0] * len(gap_tokens) + list(current_logprobs)
                else:
                    merged_logprobs = None

                collapsed_step.step_data = ProcessedStepData(
                    prompt=prefix_data.prompt,
                    response=new_response,
                    mask=new_mask,
                    step_reward=step.reward,
                    step_id=prefix_data.step_id,
                    multi_modal_inputs=step.model_output.multi_modal_inputs or {},
                    advantage=step.advantage,
                    logprobs=merged_logprobs,
                )
                last_modified_idx = i
                merged = True
                break

        if not merged:
            processed_steps.append(
                CollapsedTrajectoryStep(
                    step_data=ProcessedStepData(
                        prompt=prompt_ids,
                        response=response_ids,
                        mask=torch.ones(len(response_ids), dtype=torch.long),
                        step_reward=step.reward,
                        step_id=f"{trajectory_id}_step{step_idx}",
                        multi_modal_inputs=step.model_output.multi_modal_inputs or {},
                        advantage=step.advantage,
                        logprobs=step.model_output.logprobs,
                    ),
                    trajectory_id=trajectory_id,
                    traj_reward=traj_reward,
                    step_num=n_steps,
                    is_last_step=False,
                    group_role=name,
                )
            )
            last_modified_idx = len(processed_steps) - 1

    if processed_steps and last_modified_idx >= 0:
        processed_steps[last_modified_idx].is_last_step = True

    if processed_steps and len(processed_steps) < n_steps:
        logger.info(f"[collapse] {trajectory_id}: {n_steps} steps -> {len(processed_steps)} rows")

    return processed_steps
