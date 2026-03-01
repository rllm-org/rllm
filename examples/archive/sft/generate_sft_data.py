import argparse
import asyncio
import os

import pandas as pd

from rllm.agents.agent import Trajectory
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer


def load_problems(num_samples, trajectories_per_problem=4):
    dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    if dataset is None:
        raise RuntimeError("Dataset 'deepscaler_math' not found. Run prepare_math_data.py first to register the dataset.")

    data = dataset.get_data()
    if num_samples < len(data):
        df = pd.DataFrame(data)
        df = df.sample(n=num_samples, random_state=42)
        data = df.to_dict("records")

    # Generate multiple trajectories per problem
    tasks = []
    for i in range(trajectories_per_problem):
        for j, row in enumerate(data):
            tasks.append({"question": row["question"], "ground_truth": row["ground_truth"], "uid": f"{j}_{i}"})

    return tasks


async def generate_trajectories(tasks) -> list[Trajectory]:
    """Generate trajectories using Qwen3-4B with tool usage."""
    from transformers import AutoTokenizer

    from rllm.agents import ToolAgent
    from rllm.engine import AsyncAgentExecutionEngine
    from rllm.environments.tools.tool_env import ToolEnvironment
    from rllm.rewards.reward_fn import math_reward_fn

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model_name = "Qwen/Qwen3-4B"

    agent_args = {"tools": ["python"], "parser_name": "qwen", "system_prompt": "You are a math assistant that can write python to solve math problems."}
    env_args = {"tools": ["python"], "reward_fn": math_reward_fn}
    engine = AsyncAgentExecutionEngine(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        sampling_params={"temperature": 0.6, "top_p": 0.95, "model": model_name},
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None"},
        max_response_length=15000,
        max_prompt_length=2048,
        n_parallel_agents=64,
    )

    return await engine.execute_tasks(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--trajectories_per_problem", type=int, default=4)
    parser.add_argument("--reward_threshold", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="sft_data.parquet")
    args = parser.parse_args()

    # Load problems with replication
    tasks = load_problems(args.num_samples, args.trajectories_per_problem)

    print(f"Generating {len(tasks)} trajectories from {args.num_samples} problems ({args.trajectories_per_problem} per problem)...")

    # Generate trajectories
    results = asyncio.run(generate_trajectories(tasks))

    # Process trajectories with tool call filtering
    sft_data = AgentSFTTrainer.process_trajectories(results, args.reward_threshold, filter_tool_calls=True)

    # Save results
    if sft_data:
        pd.DataFrame(sft_data).to_parquet(args.output, index=False)
        lengths = [len(" ".join([m["content"] for m in ex["messages"] if m["role"] == "assistant"])) for ex in sft_data]
        print(f"Saved {len(sft_data)} examples with tool usage. Response lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) // len(lengths)}")
    else:
        print("No valid data generated!")


if __name__ == "__main__":
    main()
