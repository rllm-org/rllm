"""
Test script for SkyRL backend with solver-judge workflow.
"""

import hydra

from examples.solver_judge.solver_judge_flow import SolverJudgeWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.experimental.skyrl.skyrl_launcher import SkyRLTrainerLauncher
from rllm.rewards.countdown_reward import countdown_reward_fn


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")
    val_max_samples = int(config.trainer.get("val_max_samples", -1))
    if test_dataset is not None and val_max_samples > 0:
        test_dataset = test_dataset.select(range(min(val_max_samples, len(test_dataset))))

    trainer = SkyRLTrainerLauncher(
        workflow_class=SolverJudgeWorkflow,
        workflow_args={
            "n_solutions": 2,
            "reward_function": countdown_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
