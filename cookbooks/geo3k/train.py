"""Train geo3k VLM geometry solver using the Python API.

Usage (from rllm repo root):
    python cookbooks/geo3k/train.py

Or with Hydra overrides:
    python cookbooks/geo3k/train.py model.name=Qwen/Qwen3-VL-30B-A3B-Instruct training.group_size=4
"""

import hydra
from evaluator import geo3k_evaluator
from geo3k_flow import geo3k_flow
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("geo3k", "train")
    test_dataset = DatasetRegistry.load_dataset("geo3k", "test")

    if train_dataset is None:
        raise RuntimeError("geo3k train split not found. Run: rllm dataset pull geo3k")

    trainer = AgentTrainer(
        backend="tinker",
        agent_flow=geo3k_flow,
        evaluator=geo3k_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
