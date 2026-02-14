import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

from .fin_qa_agent import FinQAAgent
from .fin_qa_environment import FinQAEnvironment
from .train_finqa import FinQAWorkflow


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="tinker_rl_trainer",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("finqa", "train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "val")

    trainer = AgentTrainer(
        workflow_class=FinQAWorkflow,
        workflow_args={
            "agent_cls": FinQAAgent,
            "env_cls": FinQAEnvironment,
            "max_steps": 20,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="tinker",
    )
    trainer.train()


if __name__ == "__main__":
    main()
