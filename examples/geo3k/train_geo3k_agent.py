#!/usr/bin/env python3
"""
Training script for GEO3K multimodal geometry agent using RLLM framework.

This script follows RLLM patterns and leverages Verl's multimodal capabilities
for training on geometry problems with images.
"""

import hydra
from omegaconf import OmegaConf, open_dict

from rllm.agents import Geo3kAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="geo3k_multimodal_trainer", version_base=None)
def main(config):
    """
    Main training function for GEO3K multimodal agent.

    This follows the same pattern as other RLLM examples while enabling
    multimodal training through Verl's capabilities.
    """
    # Load datasets (must be prepared first using prepare_geo3k_data.py)
    train_dataset = DatasetRegistry.load_dataset("geo3k_train", "train")
    test_dataset = DatasetRegistry.load_dataset("geo3k_test", "test")

    # Ensure configuration matches Verl GEO3K example defaults
    if hasattr(config, "actor_rollout_ref") and hasattr(config.actor_rollout_ref, "model"):
        with open_dict(config.actor_rollout_ref.model):
            config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-VL-7B-Instruct"

    if not hasattr(config, "multimodal") or config.multimodal is None:
        with open_dict(config):
            config.multimodal = OmegaConf.create({"enable": True, "image_key": "images"})
    else:
        with open_dict(config.multimodal):
            config.multimodal.enable = True
            config.multimodal.image_key = "images"

    rejection_multiplier = 1
    if hasattr(config, "rllm") and hasattr(config.rllm, "rejection_sample") and config.rllm.rejection_sample is not None:
        rejection_multiplier = getattr(config.rllm.rejection_sample, "multiplier", 1) or 1

    if hasattr(config, "data"):
        with open_dict(config.data):
            config.data.image_key = "images"
            config.data.return_multi_modal_inputs = True
            config.data.train_batch_size = 32
            config.data.val_batch_size = 32
            config.data.gen_batch_size = 32 * max(1, rejection_multiplier)

    # Ensure PPO mini-batch sizes do not exceed data batch sizes
    train_batch_size = getattr(config.data, "train_batch_size", None)
    val_batch_size = getattr(config.data, "val_batch_size", None)

    if hasattr(config, "actor_rollout_ref") and hasattr(config.actor_rollout_ref, "actor"):
        with open_dict(config.actor_rollout_ref.actor):
            mini_batch = getattr(config.actor_rollout_ref.actor, "ppo_mini_batch_size", None)
            if train_batch_size is not None and (mini_batch is None or mini_batch > train_batch_size):
                config.actor_rollout_ref.actor.ppo_mini_batch_size = train_batch_size

    if hasattr(config, "critic"):
        with open_dict(config.critic):
            mini_batch = getattr(config.critic, "ppo_mini_batch_size", None)
            if val_batch_size is not None and (mini_batch is None or mini_batch > val_batch_size):
                config.critic.ppo_mini_batch_size = val_batch_size

    # Disable workflow mode for single-turn geo3k training
    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        with open_dict(config.rllm.workflow):
            config.rllm.workflow.use_workflow = False

    # Reduce rollout parallelism to avoid GPU OOM
    if hasattr(config, "actor_rollout_ref") and hasattr(config.actor_rollout_ref, "rollout"):
        with open_dict(config.actor_rollout_ref.rollout):
            config.actor_rollout_ref.rollout.n = 1
            config.actor_rollout_ref.rollout.gpu_memory_utilization = min(0.6, config.actor_rollout_ref.rollout.gpu_memory_utilization)
            config.actor_rollout_ref.rollout.max_num_batched_tokens = 4096
            config.actor_rollout_ref.rollout.max_num_seqs = 128

        if hasattr(config.actor_rollout_ref.rollout, "agent"):
            with open_dict(config.actor_rollout_ref.rollout.agent):
                current_workers = getattr(config.actor_rollout_ref.rollout.agent, "num_workers", 0) or 0
                config.actor_rollout_ref.rollout.agent.num_workers = max(1, current_workers)

    # Agent configuration - minimal args following RLLM patterns
    agent_args = {
        "accumulate_thinking": True,
        "include_images_in_completion": True,  # Enable image info in completions for multimodal training
    }

    # Environment configuration - simple single-turn math environment
    env_args = {
        "reward_fn": math_reward_fn,
    }

    # Initialize trainer following RLLM patterns
    trainer = AgentTrainer(
        agent_class=Geo3kAgent,
        env_class=SingleTurnEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
