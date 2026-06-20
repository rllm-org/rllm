"""Train a migration agent on MigrationBench using Verl backend + AgentCore remote runtime.

The agent runs inside an AgentCore container and migrates Java 8→17 repos.
Build/deploy the agent and upload the dataset to S3 by following:
    https://github.com/awslabs/agentcore-rl-toolkit/tree/main/examples/strands_migration_agent

Usage (from this cookbook folder, cd cookbooks/migrationbench):
    # 1. Build + deploy the agent and upload data (see strands_migration_agent README).
    # 2. Register the dataset locally from the data S3 bucket:
    python prepare_migrationbench_data.py --s3-bucket-name <your-bucket>

    # 3. Then train:
    bash train_agentcore_migrationbench_verl.sh
"""

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("migration_bench", "train")
    test_dataset = DatasetRegistry.load_dataset("migration_bench", "test")

    trainer = AgentTrainer(
        backend="verl",
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
