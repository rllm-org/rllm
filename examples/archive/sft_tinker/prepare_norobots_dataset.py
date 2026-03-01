"""
Prepare NoRobots dataset for rLLM Tinker SFT training.

This script downloads the HuggingFaceH4/no_robots dataset and registers it
with rLLM's DatasetRegistry, matching the RL trainer's dataset handling.
"""

import argparse

import datasets

from rllm.data.dataset import DatasetRegistry


def prepare_norobots_dataset():
    """
    Download and prepare the NoRobots dataset using DatasetRegistry.

    This replicates the dataset setup from tinker-cookbook's sl_basic.py
    which uses HuggingFaceH4/no_robots, but uses rLLM's DatasetRegistry
    similar to the RL trainer.
    """
    print("Loading NoRobots dataset from HuggingFace...")
    dataset = datasets.load_dataset("HuggingFaceH4/no_robots")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Shuffle train dataset (matching tinker-cookbook)
    train_dataset = train_dataset.shuffle(seed=0)

    # Register datasets with rLLM's DatasetRegistry
    print("\nRegistering datasets with DatasetRegistry...")
    train_dataset = DatasetRegistry.register_dataset("norobots", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("norobots", test_dataset, "test")

    print("\n✓ Dataset registered successfully!")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    print("\nYou can now run:")
    print("  ./train_norobots_tinker.sh")

    # Show a sample
    print("\nSample conversation:")
    sample = train_dataset[0]
    print(f"Messages: {sample['messages']}")

    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NoRobots dataset for rLLM")
    args = parser.parse_args()

    train_dataset, test_dataset = prepare_norobots_dataset()
    print("\nDatasets ready:")
    print(f"  Train: {train_dataset}")
    print(f"  Test: {test_dataset}")
