import hydra
from omegaconf import DictConfig

from examples.solver_judge_distill.simple_math_workflow import SimpleMathWorkflow, math_reward_fn
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="tinker_rl_trainer")
def main(config: DictConfig):
    """Main training function for simple math distillation."""
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("deepmath_opd", "train")
    test_dataset = DatasetRegistry.load_dataset("deepmath_opd", "test")

    # Create trainer with SimpleMathWorkflow (1 trajectory per episode)
    trainer = AgentTrainer(
        workflow_class=SimpleMathWorkflow,
        workflow_args={
            "reward_function": math_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )

    trainer.train()


if __name__ == "__main__":
    main()
