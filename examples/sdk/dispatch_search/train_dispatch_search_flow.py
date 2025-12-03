import hydra

from examples.sdk.dispatch_search.dispatch_search_workflow import DispatcherSearcherWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("dispatch_search", "train")
    test_dataset = DatasetRegistry.load_dataset("dispatch_search", "test")

    trainer = AgentTrainer(
        workflow_class=DispatcherSearcherWorkflow,
        workflow_args={
            "top_k": 3,
            "shuffle_retrieved_info": False,
            "effort_param": 0.5,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
