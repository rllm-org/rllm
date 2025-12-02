import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.eval_protocol_workflow import EvalProtocolWorkflow


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("frozen_lake_eval_protocol", "train")
    test_dataset = DatasetRegistry.load_dataset("frozen_lake_eval_protocol", "test")

    trainer = AgentTrainer(
        workflow_class=EvalProtocolWorkflow,
        workflow_args={
            "env_path": "eval_protocol.benchmarks.test_frozen_lake",
            "lite_llm_prefix": "fireworks_ai/",
            "steps": 30,
            "temperature": 1.0,
            "max_tokens": 32768,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="fireworks",
    )
    trainer.train()


if __name__ == "__main__":
    main()
