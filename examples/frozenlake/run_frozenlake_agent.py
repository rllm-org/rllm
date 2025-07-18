import asyncio

from transformers import AutoTokenizer

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.utils import compute_pass_at_k


def load_frozenlake_data():
    if DatasetRegistry.dataset_exists("frozenlake", "test"):
        test_dataset = DatasetRegistry.load_dataset("frozenlake", "test")
        return test_dataset.get_data()

    print("FrozenLake datasets not found. Preparing datasets...")
    from prepare_frozenlake_data import prepare_frozenlake_data

    train_dataset, test_dataset = prepare_frozenlake_data()

    return test_dataset.get_data()


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 256

    model_name = "internlm/internlm2_5-1_8b-chat"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    agent_args = {
        "max_steps": 10,
        "use_accumulate_history": True,
    }

    env_args = {
        "max_steps": 8,
        "is_slippery": False,
    }

    engine = AgentExecutionEngine(
        agent_class=FrozenLakeAgent,
        env_class=FrozenLakeEnv,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    tasks = load_frozenlake_data()

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
