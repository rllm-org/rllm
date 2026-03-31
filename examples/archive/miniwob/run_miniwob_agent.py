import argparse
import asyncio

from transformers import AutoTokenizer

from rllm.agents.miniwob_agent import MiniWobAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.browsergym.browsergym import BrowserGymEnv
from rllm.utils import compute_pass_at_k


def load_miniwob_data():
    if DatasetRegistry.dataset_exists("miniwob", "test"):
        test_dataset = DatasetRegistry.load_dataset("miniwob", "test")
        return test_dataset.get_data()

    print("MiniWoB datasets not found. Preparing datasets...")
    from prepare_miniwob_data import prepare_miniwob_data

    train_dataset, test_dataset = prepare_miniwob_data()

    return test_dataset.get_data()


if __name__ == "__main__":
    import os

    url = os.getenv("MINIWOB_URL")
    if url is None:
        raise Exception("MINIWOB_URL is not set.")
    else:
        print(f"MINIWOB_URL is set to: {url}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--stepwise", action="store_true", help="Run in stepwise mode")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 256

    model_name = "Qwen/Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    env_args = {
        "subtask": "miniwob",
        "miniwob_url": url,
    }

    # Set parameters based on stepwise flag
    if not args.stepwise:
        agent_args = {
            "use_accumulate_thinking": True,
            "use_full_conversation": True,
        }
        max_prompt_length = 3072
        max_response_length = 16384
        enforce_max_prompt_length = False
    else:
        agent_args = {
            "use_accumulate_thinking": False,
            "use_full_conversation": False,
        }
        max_prompt_length = 16384
        max_response_length = 3072
        enforce_max_prompt_length = True

    engine = AgentExecutionEngine(
        agent_class=MiniWobAgent,
        env_class=BrowserGymEnv,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=max_response_length,
        max_prompt_length=max_prompt_length,
        n_parallel_agents=n_parallel_agents,
        disable_thinking=False,
        enforce_max_prompt_length=enforce_max_prompt_length,
    )

    tasks = load_miniwob_data()

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
