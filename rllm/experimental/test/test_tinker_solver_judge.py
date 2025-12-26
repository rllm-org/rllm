"""
Adapted from `examples.solver_judge_tinker.train_solver_judge_flow_tinker` to
test the unified trainer with Tinker backend.
"""

import hydra

from examples.solver_judge.solver_judge_flow import SolverJudgeWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.rewards.countdown_reward import countdown_reward_fn


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    trainer = AgentTrainer(
        workflow_class=SolverJudgeWorkflow,
        workflow_args={
            "n_solutions": 2,
            "reward_function": countdown_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend=config.rllm.backend,
    )
    trainer.train()


if __name__ == "__main__":
    main()
