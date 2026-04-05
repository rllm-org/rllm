import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.workflows.simple_workflow import SimpleWorkflow


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    trainer = AgentTrainer(
        workflow_class=SimpleWorkflow,
        workflow_args={
            "reward_function": countdown_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="verl",
    )
    trainer.train()


if __name__ == "__main__":
    main()
