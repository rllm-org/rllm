import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer

TRAIN_DATASET = "swesmith"
VAL_DATASET = "swebench-verified"


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET, "default")
    val_dataset = DatasetRegistry.load_dataset(VAL_DATASET, "default")

    if train_dataset is None:
        raise RuntimeError(f"Dataset '{TRAIN_DATASET}' not found. Run: rllm dataset pull harbor:{TRAIN_DATASET}")
    if val_dataset is None:
        raise RuntimeError(f"Dataset '{VAL_DATASET}' not found. Run: rllm dataset pull harbor:{VAL_DATASET}")

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="tinker",
    )
    trainer.train()


if __name__ == "__main__":
    main()
