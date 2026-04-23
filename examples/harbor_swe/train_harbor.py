import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("swesmith_harbor", "train")
    val_dataset = DatasetRegistry.load_dataset("swebench_verified_harbor", "test")

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="tinker",
    )
    trainer.train()


if __name__ == "__main__":
    main()
