import hydra

from examples.geo3k.geo3k_workflow import Geo3KWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import f1_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("latex_ocr", "train")
    test_dataset = DatasetRegistry.load_dataset("latex_ocr", "test")

    trainer = AgentTrainer(
        workflow_class=Geo3KWorkflow,
        workflow_args={
            "reward_function": f1_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
