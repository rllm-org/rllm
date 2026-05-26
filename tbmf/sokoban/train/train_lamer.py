"""Train the LaMer (Meta-RL) Sokoban agent using the Python API.

Usage from the rllm repo root::

    python3 -m tbmf.sokoban.train.train_lamer rllm/backend=verl
"""

import hydra
from omegaconf import DictConfig

try:
    from ..eval.sokoban_lamer_eval import sokoban_lamer_evaluator
    from ..flow.sokoban_lamer_flow import sokoban_lamer_flow
except (ImportError, ValueError):
    from eval.sokoban_lamer_eval import sokoban_lamer_evaluator
    from flow.sokoban_lamer_flow import sokoban_lamer_flow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("sokoban", "train")
    val_dataset = DatasetRegistry.load_dataset("sokoban", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("Sokoban dataset not found. Run: python3 tbmf/sokoban/prepare_sokoban_data.py")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=sokoban_lamer_flow,
        evaluator=sokoban_lamer_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
