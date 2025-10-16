import hydra

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("gsm8k", "train")
    test_dataset = DatasetRegistry.load_dataset("gsm8k", "test")

    env_args = {"reward_fn": math_reward_fn}

    # some helpful warnings when LoRA is turned on
    lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_rank > 512:
        print("WARNING: LoRA rank is greater than 512, which is not supported if you use vLLM.")
        config.actor_rollout_ref.rollout.lora_rank = 512

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
