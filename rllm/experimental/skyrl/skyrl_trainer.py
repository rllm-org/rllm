"""
SkyRL Trainer for rLLM workflows integration (Legacy).

This trainer uses rLLM's UnifiedWorkflowEngine to execute workflows with SkyRL's native training loop.
The key integration point is the generate() method which sets up episode logging metadata.
# TODO: We can completely remove this implementation.

Explanation:
    SkyrlTrainer is used when you want to use SkyRL's native training loop (RayPPOTrainer.train())
    with rLLM workflows. It extends RayPPOTrainer and overrides the generate() method to use
    RLLMGenerator, which wraps UnifiedWorkflowEngine.

    It's NOT following the unified trainer pattern.
"""

import torch
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import validate_generator_output


class SkyrlTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: Base class already builds dataloaders, but we can override if needed for reproducibility
        # For now, we use the base class dataloaders

    @torch.no_grad()
    async def generate(
        self,
        input_batch: GeneratorInput,
    ) -> GeneratorOutput:
        """
        Generate rollouts using RLLMGenerator with UnifiedWorkflowEngine.

        If colocate_all is enabled:
        - before calling this method, the policy model should be on CPU and inference engine should
            be awake (i.e. on GPU).
        - after calling this method, the same model placement still holds.
        """
        # Initialize UnifiedWorkflowEngine pool if generator uses it and pool is not initialized
        # RLLMGenerator exposes workflow_engine attribute
        if hasattr(self.generator, "workflow_engine"):
            if self.generator.workflow_engine.workflow_queue is None:
                await self.generator.workflow_engine.initialize_pool()

            # Set training step for episode logging (rLLM abstraction)
            # Calculate epoch from global_step and dataloader length
            batch_metadata = input_batch.get("batch_metadata")
            if batch_metadata:
                global_step = batch_metadata.global_step if hasattr(batch_metadata, "global_step") else self.global_step
                training_phase = batch_metadata.training_phase if hasattr(batch_metadata, "training_phase") else "train"
            else:
                global_step = self.global_step
                training_phase = "train"

            # Calculate epoch: epoch = global_step // steps_per_epoch
            # Note: global_step starts at 1, so we subtract 1 before dividing
            steps_per_epoch = len(self.train_dataloader) if self.train_dataloader else 1
            epoch = (global_step - 1) // steps_per_epoch if global_step > 0 else 0

            self.generator.workflow_engine.set_training_step(global_step, mode=training_phase, epoch=epoch)

        # NOTE: we assume that .generate returns samples in the same order as passed in
        # Here RLLMGenerator would return output from UnifiedWorkflowEngine
        generator_output: GeneratorOutput = await self.generator.generate(input_batch)

        # add rollout metrics to self.all_metrics
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])

        # Validate output - use actual number of responses (some prompts may be filtered out if workflows failed)
        # The generator filters out prompts that don't have valid responses, so we validate against the actual response count
        num_responses = len(generator_output["response_ids"])
        validate_generator_output(num_responses, generator_output)

        return generator_output
