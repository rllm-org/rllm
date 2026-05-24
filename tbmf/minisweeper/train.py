"""Train the MiniSweeper agent using the Python API.

Usage from the rllm repo root::

    python3 tbmf/minisweeper/train.py rllm/backend=tinker
"""

import hydra
from omegaconf import DictConfig

try:
    from .minisweeper_eval import minisweeper_evaluator
    from .minisweeper_flow import minisweeper_flow
except ImportError:
    from minisweeper_eval import minisweeper_evaluator
    from minisweeper_flow import minisweeper_flow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("minisweeper", "train")
    val_dataset = DatasetRegistry.load_dataset("minisweeper", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError(
            "MiniSweeper dataset not found. Run: python3 tbmf/minisweeper/prepare_minisweeper_data.py"
        )

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=minisweeper_flow,
        evaluator=minisweeper_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
