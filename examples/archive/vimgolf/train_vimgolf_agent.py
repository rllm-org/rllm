import hydra
from lib import VimGolfSingleTurnAgent, VimGolfSingleTurnEnv

from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    dataset = DatasetRegistry.load_dataset(name="vimgolf-public-challenges", split="train")

    trainer = AgentTrainer(
        agent_class=VimGolfSingleTurnAgent,
        env_class=VimGolfSingleTurnEnv,
        config=config,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
