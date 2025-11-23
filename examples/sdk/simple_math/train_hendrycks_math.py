import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.sdk.shortcuts import get_chat_client
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    assert train_dataset, "Train dataset not found. Please run examples/simple_math/prepare_math_dataset.py first."
    assert test_dataset, "Test dataset not found. Please run examples/simple_math/prepare_math_dataset.py first."

    # Define run function that recreates the client inside to avoid closure capture
    # This ensures the function is fully serializable for Ray
    def rollout(**kwargs):
        # Recreate the client inside the function to avoid serialization issues
        # This ensures the function doesn't capture non-serializable objects
        ground_truth = kwargs["ground_truth"]
        question = kwargs["question"]
        client = get_chat_client(base_url="http://localhost:4000/v1", api_key="EMPTY")
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            messages=[
                {"role": "user", "content": question},
            ],
        )
        response_text = response.choices[0].message.content
        reward = math_reward_fn({"response": response_text, "ground_truth": ground_truth}, response_text).reward
        return reward * 1.0

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        agent_run_func=rollout,
    )
    trainer.train()


if __name__ == "__main__":
    main()
