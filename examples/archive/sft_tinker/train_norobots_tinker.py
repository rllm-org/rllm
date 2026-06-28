"""Train on the NoRobots dataset using the Tinker SFT backend.

Replicates tinker-cookbook's sl_basic.py setup (Llama-3.1-8B, no_robots,
ALL_ASSISTANT_MESSAGES) via the unified SFT trainer.

Usage:
    # First, prepare the dataset:
    python prepare_norobots_dataset.py

    # Then train:
    python train_norobots_tinker.py
"""

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
from rllm.trainer.sft import SFTSpec


def main():
    train_dataset = DatasetRegistry.load_dataset("norobots", "train")
    val_dataset = DatasetRegistry.load_dataset("norobots", "test")

    if train_dataset is None or val_dataset is None:
        raise ValueError("Datasets not found! Run prepare_norobots_dataset.py first:\n  python -m examples.archive.sft_tinker.prepare_norobots_dataset")

    spec = SFTSpec(
        model="meta-llama/Llama-3.1-8B",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=2e-4,
        lr_schedule="linear",
        batch_size=128,
        max_length=32768,
        tokenize_method="cumulative",
        project="rllm-tinker-sft",
        experiment="norobots",
    )

    AgentSFTTrainer(spec, backend="tinker").train()


if __name__ == "__main__":
    main()
