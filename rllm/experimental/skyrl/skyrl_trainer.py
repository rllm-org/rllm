"""
SkyRL Trainer for rLLM workflows integration.

This trainer uses rLLM's UnifiedWorkflowEngine to execute workflows with SkyRL's training infrastructure.
The key integration point is the generate() method which sets up episode logging metadata.

Usage:
    SkyrlTrainer is used when you want to use SkyRL's native training loop (RayPPOTrainer.train())
    with rLLM workflows. It extends RayPPOTrainer and overrides the generate() method to use
    RLLMGenerator, which wraps UnifiedWorkflowEngine.

    This is different from SkyRLBackend, which is used with the unified trainer pattern.
    SkyRLBackend uses TrainingInputBatch and follows the BackendProtocol interface.

    When to use SkyrlTrainer:
    - You want to use SkyRL's native training loop (BasePPOExp pattern)
    - You want to use rLLM workflows for episode generation
    - You don't need the unified trainer's rejection sampling or other features

    When to use SkyRLBackend:
    - You want to use the unified trainer pattern
    - You need rejection sampling, trajectory grouping, etc.
    - You want backend-agnostic training code
"""

import asyncio
import torch

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from skyrl_train.utils.trainer_utils import validate_generator_output


class SkyrlTrainer(RayPPOTrainer):
    """PPO Trainer for rLLM workflows using SkyRL infrastructure.
    
    This trainer integrates rLLM's UnifiedWorkflowEngine with SkyRL's training pipeline.
    The generator (RLLMGenerator) uses UnifiedWorkflowEngine to execute workflows and
    transform results to SkyRL's GeneratorOutput format.
    
    This trainer is used with SkyRL's native training loop (BasePPOExp pattern).
    It extends RayPPOTrainer and overrides the generate() method to use RLLMGenerator.
    
    For advantage computation, this trainer uses the standard advantage estimators
    from skyrl_train.utils.ppo_utils via the AdvantageEstimatorRegistry. The advantage
    estimator is specified in the config (e.g., "grpo", "rloo", "gae", "reinforce++").
    
    Example usage:
        from rllm.experimental.skyrl.skyrl_trainer import SkyrlTrainer
        from rllm.experimental.skyrl.rllm_generator import RLLMGenerator
        from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
        
        # Create UnifiedWorkflowEngine
        workflow_engine = UnifiedWorkflowEngine(...)
        
        # Create RLLMGenerator
        generator = RLLMGenerator(
            workflow_engine=workflow_engine,
            tokenizer=tokenizer,
            max_response_length=4096,
        )
        
        # Create trainer
        trainer = SkyrlTrainer(
            cfg=config,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )
        
        # Train using SkyRL's native loop
        trainer.train()
    """
    
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
        if hasattr(self.generator, 'workflow_engine'):
            if self.generator.workflow_engine.workflow_queue is None:
                await self.generator.workflow_engine.initialize_pool()
            
            # Set training step for episode logging (rLLM abstraction)
            # Calculate epoch from global_step and dataloader length
            batch_metadata = input_batch.get("batch_metadata")
            if batch_metadata:
                global_step = batch_metadata.global_step if hasattr(batch_metadata, 'global_step') else self.global_step
                training_phase = batch_metadata.training_phase if hasattr(batch_metadata, 'training_phase') else "train"
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

        # Validate output - base function takes num_prompts, not input_batch
        validate_generator_output(len(input_batch["prompts"]), generator_output)

        return generator_output

