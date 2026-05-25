"""Train the LaMer (Meta-RL) MineSweeper agent."""

import hydra
from omegaconf import DictConfig

try:
    from .minisweeper_lamer_eval import minisweeper_lamer_evaluator
    from .minisweeper_lamer_flow import minisweeper_lamer_flow
except ImportError:
    from minisweeper_lamer_eval import minisweeper_lamer_evaluator
    from minisweeper_lamer_flow import minisweeper_lamer_flow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("minisweeper", "train")
    val_dataset = DatasetRegistry.load_dataset("minisweeper", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("MineSweeper dataset not found. Run: python3 tbmf/minisweeper/prepare_minisweeper_data.py")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=minisweeper_lamer_flow,
        evaluator=minisweeper_lamer_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
