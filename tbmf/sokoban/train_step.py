"""Train the step-based (non-cumulative) Sokoban agent.

Usage from the rllm repo root::

    python3 tbmf/sokoban/train_step.py rllm/backend=tinker
"""

import hydra
from omegaconf import DictConfig

try:
    from .sokoban_eval import sokoban_evaluator
    from .sokoban_step_flow import sokoban_step_flow
except ImportError:
    from sokoban_eval import sokoban_evaluator
    from sokoban_step_flow import sokoban_step_flow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("sokoban", "train")
    val_dataset = DatasetRegistry.load_dataset("sokoban", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("Sokoban dataset not found. Run: python3 tbmf/sokoban/prepare_sokoban_data.py")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=sokoban_step_flow,
        evaluator=sokoban_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
