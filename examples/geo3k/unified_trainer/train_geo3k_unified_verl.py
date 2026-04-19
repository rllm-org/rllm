import hydra

from examples.geo3k.geo3k_workflow import Geo3KWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.rewards.reward_fn import math_reward_fn


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("geo3k", "train")
    test_dataset = DatasetRegistry.load_dataset("geo3k", "test")

    trainer = AgentTrainer(
        workflow_class=Geo3KWorkflow,
        workflow_args={
            "reward_function": math_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="verl",
    )
    trainer.train()


if __name__ == "__main__":
    main()
