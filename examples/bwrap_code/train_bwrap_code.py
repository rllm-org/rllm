import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.bwrap_code_workflow import BwrapCodeWorkflow

from examples.bwrap_code.prepare_data import prepare_bwrap_code_data


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    prepare_bwrap_code_data()
    train_dataset = DatasetRegistry.load_dataset("bwrap_code", "train")
    test_dataset = DatasetRegistry.load_dataset("bwrap_code", "test")

    trainer = AgentTrainer(
        workflow_class=BwrapCodeWorkflow,
        workflow_args={"exec_timeout": 30},
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
