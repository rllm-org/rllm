"""
RLLM Generator that uses rLLM workflows for SkyRL training.

This generator implements SkyRL's GeneratorInterface and uses UnifiedWorkflowEngine
to run rLLM workflows, then converts the resulting Episodes to GeneratorOutput.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput

if TYPE_CHECKING:
    from rllm.agents.agent import Episode
    from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


class RLLMGenerator(GeneratorInterface):
    """rLLM generator that uses rLLM workflows for SkyRL training.

    This generator:
    1. Receives GeneratorInput from SkyRL trainer
    2. Uses UnifiedWorkflowEngine to run rLLM workflows and generate Episodes
    3. Converts Episodes to GeneratorOutput for SkyRL training
    """

    def __init__(
        self,
        workflow_engine: UnifiedWorkflowEngine | None = None,
        tokenizer=None,
        max_response_length: int = 4096,
    ):
        """Initialize the rLLM generator.

        Args:
            workflow_engine: UnifiedWorkflowEngine instance for running workflows.
                Can be None initially and set later (e.g., in the backend when workflow engine is available).
            tokenizer: Tokenizer instance for tokenization
            max_response_length: Maximum response length
        """
        self.workflow_engine = workflow_engine
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories using rLLM workflows via UnifiedWorkflowEngine.
        
        Args:
            input_batch: SkyRL's GeneratorInput with prompts and environment info

        Returns:
            GeneratorOutput: Tensor data ready for SkyRL training
        
        Raises:
            RuntimeError: If workflow_engine is not set.
        """
        if self.workflow_engine is None:
            raise RuntimeError("workflow_engine must be set before calling generate(). Set it via the backend or directly.")
        from rllm.agents.agent import Episode, Trajectory, Step
        from rllm.engine.rollout import ModelOutput
        from skyrl_train.generators.utils import get_rollout_metrics

        # Extract data from GeneratorInput
        prompts = input_batch["prompts"]
        env_extras_list = input_batch.get("env_extras", [])
        trajectory_ids = input_batch.get("trajectory_ids", [])
        batch_metadata = input_batch.get("batch_metadata")

        # Convert GeneratorInput to tasks for UnifiedWorkflowEngine
        # Reconstruct original rLLM format from prompts + env_extras
        tasks = []
        task_ids = []
        for i, prompt in enumerate(prompts):
            # Get env_extras for this item (if available)
            env_extras = env_extras_list[i] if i < len(env_extras_list) else {}
            
            # Reconstruct original rLLM format task
            task = {}
            
            # Check if we have original prompt info stored in env_extras
            original_prompt_key = env_extras.get("_rllm_original_prompt_key")
            original_prompt_value = env_extras.get("_rllm_original_prompt_value")
            
            if original_prompt_key and original_prompt_value is not None:
                # Restore original prompt as string in original key
                task[original_prompt_key] = original_prompt_value
            elif isinstance(prompt, list) and len(prompt) > 0:
                # Extract string from chat format: [{"role": "user", "content": "..."}]
                # Default to "question" if we can't determine the original key
                content = prompt[0].get("content", "")
                task["question"] = content
            else:
                # Fallback: use prompt as-is
                task["prompt"] = prompt
            
            # Copy all other fields from env_extras (excluding internal metadata)
            # Also exclude "messages" to avoid conflicts - we reconstruct from the original prompt key
            for key, value in env_extras.items():
                if not key.startswith("_rllm_") and key != "messages":
                    task[key] = value

            # Get task ID from trajectory_id if available
            if i < len(trajectory_ids) and trajectory_ids[i]:
                task_id = trajectory_ids[i].instance_id
            else:
                task_id = f"task_{i}"

            tasks.append(task)
            task_ids.append(task_id)

        # Execute workflows using UnifiedWorkflowEngine
        episodes: list[Episode] = await self.workflow_engine.execute_tasks(tasks, task_ids)

        # Convert Episodes to GeneratorOutput
        prompt_token_ids: list[list[int]] = []
        response_ids: list[list[int]] = []
        rewards: list[float] = []
        loss_masks: list[list[int]] = []
        stop_reasons: list[str] = []
        rollout_logprobs: list[list[float]] | None = []

        for episode in episodes:
            if not episode.trajectories:
                logger.warning(f"Episode {episode.id} has no trajectories, skipping")
                continue

            # For each trajectory in the episode
            for trajectory in episode.trajectories:
                if not trajectory.steps:
                    logger.warning(f"Trajectory {trajectory.uid} has no steps, skipping")
                    continue

                # Get prompt from first step
                first_step = trajectory.steps[0]
                if not isinstance(first_step.model_output, ModelOutput):
                    logger.warning(f"Step in trajectory {trajectory.uid} has no model_output, skipping")
                    continue

                prompt_tokens = first_step.model_output.prompt_ids
                prompt_token_ids.append(prompt_tokens)

                # Concatenate all response tokens from all steps
                response_tokens_list: list[int] = []
                response_logprobs_list: list[float] = []
                loss_mask_list: list[int] = []

                for step in trajectory.steps:
                    if not isinstance(step.model_output, ModelOutput):
                        continue

                    step_response = step.model_output.completion_ids
                    response_tokens_list.extend(step_response)

                    # Loss mask: 1 for all response tokens
                    loss_mask_list.extend([1] * len(step_response))

                    # Logprobs if available
                    if step.logprobs:
                        response_logprobs_list.extend(step.logprobs)
                    elif step.model_output.logprobs:
                        response_logprobs_list.extend(step.model_output.logprobs)
                    else:
                        response_logprobs_list.extend([0.0] * len(step_response))

                response_ids.append(response_tokens_list)
                loss_masks.append(loss_mask_list)

                # Reward: use trajectory reward if available, otherwise last step reward
                reward = trajectory.reward if trajectory.reward is not None else (
                    trajectory.steps[-1].reward if trajectory.steps else 0.0
                )
                rewards.append(reward)

                # Stop reason: use termination reason from episode
                stop_reason = episode.termination_reason.value if episode.termination_reason else "unknown"
                stop_reasons.append(stop_reason)

                # Add logprobs if we collected any
                if rollout_logprobs is not None:
                    rollout_logprobs.append(response_logprobs_list)

        # Get rollout metrics
        env_classes = [task.get("env_class", "unknown") for task in tasks]
        rollout_metrics = get_rollout_metrics(
            responses=[self.tokenizer.decode(rids) for rids in response_ids],
            rewards=rewards,
            env_metrics=None,
            env_classes=env_classes,
        )

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs if rollout_logprobs else None,
        }

        return generator_output

