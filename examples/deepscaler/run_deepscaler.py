import asyncio

from transformers import AutoTokenizer

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 64

    model_name = "internlm/internlm2_5-1_8b-chat"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reward_fn = math_reward_fn

    env_args = {
        "reward_fn": reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=32768,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_math_data import prepare_math_data

        _, test_dataset = prepare_math_data()

    tasks = test_dataset.repeat(n=16)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
