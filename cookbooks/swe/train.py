#!/usr/bin/env python3
"""Train SWE-bench agent using AgentTrainer with Tinker backend.

Uses the AgentFlow/Evaluator framework: SWEAgentFlow runs the agent,
SWEEvaluator grades the patch, and rllm's AgentTrainer handles the
GRPO training loop with gateway-mediated trace capture.

Usage:
    # First, register datasets
    python prepare_tinker_data.py --dataset swe_smith --split train
    python prepare_tinker_data.py --dataset swe_bench_multilingual --split test

    # Train
    python train.py \
        train_dataset=swe_smith \
        val_dataset=swe_bench_multilingual

    # With overrides
    python train.py \
        train_dataset=swe_smith \
        val_dataset=swe_bench_multilingual \
        model.name=Qwen/Qwen3-8B \
        training.group_size=8 \
        rllm.workflow.n_parallel_tasks=32
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig

_COOKBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_COOKBOOK_DIR))

load_dotenv(_COOKBOOK_DIR / ".env", verbose=False)

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer

from swe_agent_flow import SWEAgentFlow
from evaluator import SWEEvaluator


@hydra.main(version_base=None, config_path=".", config_name="tinker_unified")
def main(config: DictConfig):
    """Train SWE agent using AgentTrainer with Tinker backend."""
    train_dataset_name = config.get("train_dataset") or config.get("dataset", "swe_bench_multilingual")
    val_dataset_name = config.get("val_dataset") or config.get("dataset", "swe_bench_multilingual")

    print("=" * 60)
    print("SWE-bench Training (AgentFlow + AgentTrainer)")
    print("=" * 60)
    print(f"Train dataset: {train_dataset_name}")
    print(f"Val dataset: {val_dataset_name}")
    print(f"Model: {config.model.name}")
    print(f"Group size: {config.training.group_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Parallel tasks: {config.rllm.workflow.n_parallel_tasks}")
    print("=" * 60)

    # Load datasets
    train_dataset = DatasetRegistry.load_dataset(train_dataset_name)
    val_dataset = DatasetRegistry.load_dataset(val_dataset_name)

    if train_dataset is None:
        print(f"ERROR: Train dataset '{train_dataset_name}' not found!")
        print(f"Run: python prepare_tinker_data.py --dataset {train_dataset_name}")
        sys.exit(1)
    if val_dataset is None:
        print(f"ERROR: Val dataset '{val_dataset_name}' not found!")
        print(f"Run: python prepare_tinker_data.py --dataset {val_dataset_name}")
        sys.exit(1)

    print(f"\nDatasets: train={len(train_dataset)}, val={len(val_dataset)}")

    # swe: config keys match SWEAgentFlow.__init__ args exactly.
    swe_config = dict(config.get("swe", {}))
    agent_flow = SWEAgentFlow(**swe_config)
    evaluator = SWEEvaluator(
        command_timeout=swe_config.get("command_timeout", 120),
        sandbox_timeout=swe_config.get("sandbox_timeout", 3600),
        verbose=swe_config.get("verbose", False),
    )

    print("\nInitializing AgentTrainer (Tinker backend)...")
    trainer = AgentTrainer(
        backend="tinker",
        agent_flow=agent_flow,
        evaluator=evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
