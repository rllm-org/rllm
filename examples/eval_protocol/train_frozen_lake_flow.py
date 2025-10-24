import hydra

from examples.eval_protocol.frozen_lake_flow import FrozenLakeWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.pipeline_agent_trainer import PipelineAgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("frozen_lake_eval_protocol", "train")
    test_dataset = DatasetRegistry.load_dataset("frozen_lake_eval_protocol", "test")

    trainer = PipelineAgentTrainer(
        workflow_class=FrozenLakeWorkflow,
        workflow_args={
            "lite_llm_prefix": "fireworks_ai/",
            "steps": 30,
            "temperature": 1.0,
            "max_tokens": 32768,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
