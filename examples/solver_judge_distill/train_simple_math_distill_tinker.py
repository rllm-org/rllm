import hydra
from omegaconf import DictConfig

from examples.solver_judge_distill.simple_math_workflow import SimpleMathWorkflow, math_reward_fn
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


@hydra.main(version_base=None, config_path="../../rllm/trainer/config", config_name="tinker_rl_trainer")
def main(config: DictConfig):
    """Main training function for simple math distillation."""

    train_files = config.data.get("train_files") or "solver_judge_math:train"
    val_files = config.data.get("val_files") or "solver_judge_math:test"

    # Parse dataset name and split from "name:split" format
    if ":" in train_files:
        train_dataset_name, train_split = train_files.split(":", 1)
    else:
        train_dataset_name, train_split = train_files, "train"

    if ":" in val_files:
        val_dataset_name, val_split = val_files.split(":", 1)
    else:
        val_dataset_name, val_split = val_files, "test"

    # Load datasets
    train_dataset = DatasetRegistry.load_dataset(train_dataset_name, train_split)
    test_dataset = DatasetRegistry.load_dataset(val_dataset_name, val_split)

    if train_dataset is None:
        raise ValueError(
            f"Training dataset '{train_dataset_name}:{train_split}' not found!\n"
            "Please run the appropriate data prep script first:\n"
            "  python -m examples.solver_judge_distill.prepare_math_data        # for solver_judge_math\n"
            "  python -m examples.solver_judge_distill.prepare_deepmath_data    # for deepmath_opd\n"
            "  python -m examples.solver_judge_distill.prepare_openthoughts_math_data  # for openthoughts"
        )

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

