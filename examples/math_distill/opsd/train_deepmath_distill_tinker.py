import hydra
from omegaconf import DictConfig

from examples.math_distill.opsd.opsd_workflow import OPSDWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="tinker_rl_trainer")
def main(config: DictConfig):
    """Main training function for OPSD (self-distillation) on deepmath."""
    train_dataset = DatasetRegistry.load_dataset("deepmath_opd", "train")
    test_dataset = DatasetRegistry.load_dataset("deepmath_opd", "test")

    trainer = AgentTrainer(
        workflow_class=OPSDWorkflow,
        workflow_args={
            "clip_min": -5.0,
            "clip_max": 5.0,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )

    trainer.train()


if __name__ == "__main__":
    main()
