"""Train the LaMer (Meta-RL) ALFWorld agent.

Usage from the rllm repo root::

    python3 -m tbmf.alfworld.train.train_lamer rllm/backend=verl
"""

import hydra
from omegaconf import DictConfig

try:
    from ..eval.alfworld_lamer_eval import alfworld_lamer_evaluator
    from ..flow.alfworld_lamer_flow import alfworld_lamer_flow
except (ImportError, ValueError):
    from eval.alfworld_lamer_eval import alfworld_lamer_evaluator
    from flow.alfworld_lamer_flow import alfworld_lamer_flow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("alfworld", "train")
    val_dataset = DatasetRegistry.load_dataset("alfworld", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("ALFWorld dataset not found. Run: python3 tbmf/alfworld/prepare_alfworld_data.py")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=alfworld_lamer_flow,
        evaluator=alfworld_lamer_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
