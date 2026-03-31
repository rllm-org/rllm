"""Train solver-judge using the Python API.

Usage (from rllm repo root):
    python cookbooks/solver_judge_flow/train.py

Or with Hydra overrides:
    python cookbooks/solver_judge_flow/train.py model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import hydra
from evaluator import solver_judge_countdown_evaluator
from omegaconf import DictConfig
from solver_judge_flow import solver_judge_flow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    if train_dataset is None:
        raise RuntimeError("countdown train split not found. Run: rllm dataset pull countdown")

    trainer = AgentTrainer(
        backend="tinker",
        agent_flow=solver_judge_flow,
        evaluator=solver_judge_countdown_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
