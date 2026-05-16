"""Train Geo3KWorkflow via the unified trainer.

Backend is selected with the Hydra override ``rllm/backend=tinker|verl``
(see ``train_tinker.sh`` / ``train_verl.sh``). The unified trainer keeps
the legacy ``workflow_class=...`` API while routing through the
maintained backend launchers — the old
``rllm.trainer.AgentTrainer(backend="tinker")`` path has been removed.
"""

import hydra
from geo3k_flow import Geo3KWorkflow
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.rewards.reward_fn import math_reward_fn


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("geo3k", "train")
    val_dataset = DatasetRegistry.load_dataset("geo3k", "test")

    if train_dataset is None:
        raise RuntimeError("geo3k train split not found. Run: rllm dataset pull geo3k")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        workflow_class=Geo3KWorkflow,
        workflow_args={
            "reward_function": math_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
