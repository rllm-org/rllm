import hydra

from examples.solver_judge_distill.solver_judge_math_workflow import (
    SolverJudgeMathWorkflow,
    math_reward_fn,
)
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="tinker_rl_trainer", version_base=None)
def main(config):
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("solver_judge_math", "train")
    test_dataset = DatasetRegistry.load_dataset("solver_judge_math", "test")

    if train_dataset is None or test_dataset is None:
        print("Dataset not found. Please run prepare_math_data.py first:")
        print("  python -m examples.solver_judge_distill.prepare_math_data")
        return

    # Create trainer with Tinker backend
    trainer = AgentTrainer(
        workflow_class=SolverJudgeMathWorkflow,
        workflow_args={
            "n_solutions": 2,  # Number of solutions per problem
            "reward_function": math_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

