import asyncio
import os
from datetime import datetime

from transformers import AutoTokenizer

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import code_reward_fn
from rllm.utils import save_trajectories

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 64

    model_name = "internlm/internlm2_5-1_8b-chat"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reward_fn = code_reward_fn

    env_args = {
        "reward_fn": reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=CompetitionCodingAgent,
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
        max_response_length=65536,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("deepcoder", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_deepcoder_data import prepare_deepcoder_data

        _, test_dataset = prepare_deepcoder_data()

    tasks = test_dataset.get_data()

    results = asyncio.run(engine.execute_tasks(tasks))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_trajectories(results, filename=f"deepcoder_trajectories_{len(tasks)}_{timestamp}.pt")
