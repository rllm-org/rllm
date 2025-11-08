import hydra

from examples.solver_judge.solver_judge_flow import SolverJudgeWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.trainer.tinker.tinker_workflow_trainer import TinkerWorkflowTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="tinker_workflow_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    trainer = TinkerWorkflowTrainer(
        workflow_class=SolverJudgeWorkflow,
        workflow_args={
            "n_solutions": 2,
            "reward_function": countdown_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.fit_agent()


if __name__ == "__main__":
    main()
