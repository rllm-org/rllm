import asyncio
from datetime import datetime

from transformers import AutoTokenizer

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils.compute_pass_at_k import compute_pass_at_k, save_trajectories


def print_trajectory_examples(results, num_examples: int = 5, max_chars: int = 240):
    """Print a few trajectory snippets for quick inspection."""
    print("\n=== Sample trajectories ===")
    for idx, trajectory in enumerate(results[:num_examples]):
        task = trajectory.task
        question = task.get("question") if isinstance(task, dict) else str(task)
        print(f"[{idx}] reward={trajectory.reward} steps={len(trajectory.steps)} question={question}")
        print("=== Model responses ===")

        for step_idx, step in enumerate(trajectory.steps):
            response = step.model_response.replace("\n", " ").strip()
            print(f"step {step_idx}: {response}")

    print("=== End samples ===\n")


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 64

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    agent_args = {"tools": ["python"], "parser_name": "qwen", "system_prompt": "You are a math assistant that can write python to solve math problems."}
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=ToolAgent,
        agent_args=agent_args,
        env_class=ToolEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": "http://h200-013-092:8001/v1", "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=16384,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_math_data import prepare_math_data

        _, test_dataset = prepare_math_data()

    tasks = test_dataset.repeat(n=8)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_trajectories(results, save_dir="./trajectories/math_tool", filename=f"math_tool_trajectories_{len(tasks)}_{timestamp}.pt")
    print_trajectory_examples(results)
    compute_pass_at_k(results)
