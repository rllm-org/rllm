"""
Train on NoRobots dataset using Tinker SFT backend.

This example replicates the tinker-cookbook's sl_basic.py setup:
- Model: meta-llama/Llama-3.1-8B
- Dataset: HuggingFaceH4/no_robots
- Batch size: 128
- Max length: 32768
- Learning rate: 2e-4
- LR schedule: linear
- Train on: ALL_ASSISTANT_MESSAGES

Usage:
    # First, prepare the dataset:
    python prepare_norobots_dataset.py
    
    # Then train:
    python train_norobots_tinker.py
    
    # Or with custom settings:
    python train_norobots_tinker.py \
        model.name=meta-llama/Llama-3.1-8B
"""

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="tinker_sft_trainer",
    version_base=None,
)
def main(config: DictConfig):
    """
    Train on NoRobots dataset with Tinker backend.

    This replicates tinker-cookbook/recipes/sl_basic.py configuration
    and uses DatasetRegistry similar to the RL trainer.
    """

    # Load datasets from registry (similar to RL trainer)
    train_dataset = DatasetRegistry.load_dataset("norobots", "train")
    test_dataset = DatasetRegistry.load_dataset("norobots", "test")

    if train_dataset is None or test_dataset is None:
        raise ValueError("Datasets not found! Please run prepare_norobots_dataset.py first:\n  python -m examples.sft_tinker.prepare_norobots_dataset")

    # Initialize trainer with Tinker backend
    trainer = AgentSFTTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
