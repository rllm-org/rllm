"""Train the ALFWorld agent using the step-based (non-cumulative) flow.

Usage (from rllm repo root)::

    python tbmf/alfworld/train_step.py rllm/backend=tinker

    # Hydra overrides:
    python tbmf/alfworld/train_step.py model.name=Qwen/Qwen3-4B training.group_size=4
"""

import hydra
from omegaconf import DictConfig

try:
    from .alfworld_eval import alfworld_evaluator
    from .alfworld_step_flow import alfworld_step_flow
except ImportError:
    from alfworld_eval import alfworld_evaluator
    from alfworld_step_flow import alfworld_step_flow
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
        agent_flow=alfworld_step_flow,
        evaluator=alfworld_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
