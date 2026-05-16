"""Train SolverJudgeWorkflow via the unified trainer.

Backend is selected with the Hydra override ``rllm/backend=tinker|verl``
(see ``train_tinker.sh`` / ``train_verl.sh``). The unified trainer keeps
the legacy ``workflow_class=...`` API while routing through the
maintained backend launchers — the old
``rllm.trainer.AgentTrainer(backend="tinker")`` path has been removed.
"""

import hydra
from omegaconf import DictConfig
from solver_judge_flow import SolverJudgeWorkflow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.rewards.countdown_reward import countdown_reward_fn


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    val_dataset = DatasetRegistry.load_dataset("countdown", "test")

    if train_dataset is None:
        raise RuntimeError("countdown train split not found. Run: rllm dataset pull countdown")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        workflow_class=SolverJudgeWorkflow,
        workflow_args={
            "n_solutions": 2,
            "reward_function": countdown_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
