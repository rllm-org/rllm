"""Train the deepcoder agent using the Python API.

Usage (from rllm repo root)::

    python cookbooks/deepcoder/train.py rllm/backend=tinker

    # Hydra overrides:
    python cookbooks/deepcoder/train.py model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import hydra
from deepcoder_flow import deepcoder_flow
from evaluator import deepcoder_evaluator
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("deepcoder", "train")
    val_dataset = DatasetRegistry.load_dataset("deepcoder", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("Deepcoder dataset not found. Run: python cookbooks/deepcoder/prepare_data.py")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=deepcoder_flow,
        evaluator=deepcoder_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
