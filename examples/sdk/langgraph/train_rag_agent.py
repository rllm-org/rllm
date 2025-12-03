import hydra

from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

from .search_agent_langgraph import run_search_agent


def run_agent(question, ground_truth, **kwargs):
    return run_search_agent(question, ground_truth)["reward"]


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")

    # Use the registry-based approach (comment out the other approach)
    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_run_func=run_agent,
    )

    trainer.train()


if __name__ == "__main__":
    main()
