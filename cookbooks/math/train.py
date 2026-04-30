"""Train the math agent using the Python API.

Usage (from rllm repo root)::

    python cookbooks/math/train.py rllm/backend=tinker

    # Hydra overrides:
    python cookbooks/math/train.py model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import hydra
from math_eval import math_evaluator
from math_flow import math_flow
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    val_dataset = DatasetRegistry.load_dataset("math500", "test")

    if train_dataset is None:
        raise RuntimeError("hendrycks_math train split not found. Run: rllm dataset pull hendrycks_math")
    if val_dataset is None:
        raise RuntimeError("math500 test split not found. Run: rllm dataset pull math500")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=math_flow,
        evaluator=math_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
