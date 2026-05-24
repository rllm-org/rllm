"""Train the ALFWorld agent using the Python API.

Usage (from rllm repo root)::

    python tbmf/alfworld/train.py rllm/backend=tinker

    # Hydra overrides:
    python tbmf/alfworld/train.py model.name=Qwen/Qwen3-4B training.group_size=4
"""

import hydra
from omegaconf import DictConfig

from .alfworld_eval import alfworld_evaluator
from .alfworld_flow import alfworld_flow
from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("alfworld", "train")
    val_dataset = DatasetRegistry.load_dataset("alfworld", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError(
            "ALFWorld dataset not found. Run: python tbmf/alfworld/prepare_alfworld_data.py"
        )

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=alfworld_flow,
        evaluator=alfworld_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
