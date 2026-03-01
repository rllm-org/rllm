import os

import hydra

from rllm.agents.miniwob_agent import MiniWobAgent
from rllm.data import DatasetRegistry
from rllm.environments.browsergym.browsergym import BrowserGymEnv
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("miniwob", "train")
    val_dataset = DatasetRegistry.load_dataset("miniwob", "test")

    url = os.getenv("MINIWOB_URL")
    if url is None:
        raise Exception("MINIWOB_URL is not set.")
    else:
        print(f"MINIWOB_URL is set to: {url}")

    env_args = {
        "subtask": "miniwob",
        "miniwob_url": url,
    }

    trainer = AgentTrainer(
        agent_class=MiniWobAgent,
        env_class=BrowserGymEnv,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
