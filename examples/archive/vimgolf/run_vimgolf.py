import asyncio

from lib import VimGolfSingleTurnAgent, VimGolfSingleTurnEnv
from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import compute_pass_at_k


def load_vimgolf_data():
    if DatasetRegistry.dataset_exists(name="vimgolf-public-challenges", split="train"):
        test_dataset = DatasetRegistry.load_dataset(name="vimgolf-public-challenges", split="train")
        return test_dataset.get_data()
    raise ValueError("vimgolf-public-challenges dataset not found. Please run `python prepare_vimgolf_data.py` to create the dataset.")


if __name__ == "__main__":
    import argparse
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to be used for evaluation",
    )
    args = parser.parse_args()

    model_name = args.model_name  # to be passed via command line
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = {"temperature": 1, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=VimGolfSingleTurnAgent,
        env_class=VimGolfSingleTurnEnv,
        agent_args={},
        env_args={},
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        n_parallel_agents=48,
        max_response_length=65536,
        max_prompt_length=4096,
    )

    tasks = load_vimgolf_data()

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
