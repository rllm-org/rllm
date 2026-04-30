"""Train the finqa agent using the Python API.

Usage (from rllm repo root)::

    python cookbooks/finqa/train.py rllm/backend=tinker

    # Hydra overrides:
    python cookbooks/finqa/train.py model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import hydra
from finqa_eval import finqa_evaluator
from finqa_flow import finqa_flow
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("finqa", "train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "val")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("FinQA dataset not found. Run: python cookbooks/finqa/prepare_data.py")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=finqa_flow,
        evaluator=finqa_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
