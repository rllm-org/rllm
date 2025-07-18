import asyncio
import os

from dotenv import load_dotenv
from local_retrieval_tool import LocalRetrievalTool
from transformers import AutoTokenizer

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.utils import save_trajectories


def load_search_data(train_size=3000, test_size=100):
    """
    Load search data, preparing it if not already available.
    Returns the test dataset data for evaluation.
    """
    test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    if test_dataset is None:
        print("Dataset not found, preparing search dataset...")
        from prepare_hotpotqa_data import prepare_hotpotqa_data

        _, test_dataset = prepare_hotpotqa_data(train_size=train_size, test_size=test_size)

    return test_dataset.get_data()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if "RETRIEVAL_SERVER_URL" not in os.environ:
        os.environ["RETRIEVAL_SERVER_URL"] = "http://127.0.0.1:8000"

    load_dotenv()

    n_parallel_agents = 64

    model_name = "internlm/internlm2_5-1_8b-chat"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    tool_map = {"local_search": LocalRetrievalTool}

    engine = AgentExecutionEngine(
        agent_class=ToolAgent,
        agent_args={"tool_map": tool_map, "system_prompt": SEARCH_SYSTEM_PROMPT, "parser_name": "qwen"},
        env_class=ToolEnvironment,
        env_args={"tool_map": tool_map, "reward_fn": search_reward_fn},
        rollout_engine=None,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        config=None,
        n_parallel_agents=n_parallel_agents,
    )

    tasks = load_search_data()

    results = asyncio.run(engine.execute_tasks(tasks))

    save_trajectories(results, filename="search_trajectories.pt")
