"""
Train a math agent using TinkerAgentTrainer.

This version uses TinkerAgentTrainer which internally uses the separated
architecture (TinkerTrajectoryGenerator + TinkerPolicyTrainer) while providing
a simplified API similar to the original trainer.
"""

import hydra
from omegaconf import DictConfig

from examples.math_tinker.math_agent_with_fewshot import MathAgentWithFewshot
from examples.math_tinker.math_reward import math_reward_fn
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.trainer import AgentTrainer


@hydra.main(version_base=None, config_path="../../rllm/trainer/config", config_name="tinker_rl_trainer")
def main(config: DictConfig):
    """
    Main training function using TinkerAgentTrainer.

    Args:
        config: Hydra configuration
    """
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("gsm8k", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    if train_dataset is None or test_dataset is None:
        raise ValueError("Datasets not found! Please run prepare_tinker_math_dataset.py first:\n  python -m examples.math_tinker.prepare_tinker_math_dataset")

    # Create trainer (uses separated components internally)
    trainer = AgentTrainer(
        config=config,
        agent_class=MathAgentWithFewshot,
        env_class=SingleTurnEnvironment,
        agent_args={"use_fewshot": True},
        env_args={"reward_fn": math_reward_fn},
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )

    # Train (all orchestration handled internally by TinkerAgentTrainer)
    trainer.train()


if __name__ == "__main__":
    main()
