"""
Train a math agent using TinkerAgentTrainer.

This version uses TinkerAgentTrainer which internally uses the separated
architecture (TinkerTrajectoryGenerator + TinkerPolicyTrainer) while providing
a simplified API similar to the original trainer.
"""

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

from examples.math_tinker.math_agent_with_fewshot import MathAgentWithFewshot
from examples.math_tinker.math_reward import math_reward_fn
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.trainer.tinker.tinker_agent_trainer import TinkerAgentTrainer


class SimpleDataLoader:
    """Simple reusable dataloader."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i : i + self.batch_size]


def create_dataloader(dataset, batch_size):
    """Create a simple reusable dataloader from dataset."""
    return SimpleDataLoader(dataset, batch_size)


@hydra.main(version_base=None, config_path="../../rllm/trainer/config", config_name="tinker_agent_trainer")
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
        raise ValueError("Datasets not found! Please run prepare_math_dataset_fixed.py first:\n  python -m examples.simple_math_tinker.prepare_tinker_math_dataset")

    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, config.data.train_batch_size)
    test_dataloader = create_dataloader(test_dataset, config.data.val_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(config.tinker.model.name)
    # Create trainer (uses separated components internally)
    trainer = TinkerAgentTrainer(
        config=config,
        tokenizer=tokenizer,
        agent_class=MathAgentWithFewshot,
        env_class=SingleTurnEnvironment,
        agent_args={"use_fewshot": True},
        env_args={"reward_fn": math_reward_fn},
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
    )

    # Train (all orchestration handled internally by TinkerAgentTrainer)
    trainer.fit_agent()


if __name__ == "__main__":
    main()
