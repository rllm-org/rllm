import hydra

from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

from .search_agent_langgraph import run_search_agent


async def run_agent(question, ground_truth, **kwargs):
    try:
        result = await run_search_agent(question, ground_truth)
    except Exception:
        return 0.0
    return result["reward"]


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa-small", "test")

    # # Use the registry-based approach (comment out the other approach)
    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_run_func=run_agent,
    )

    trainer.train()


if __name__ == "__main__":
    main()
