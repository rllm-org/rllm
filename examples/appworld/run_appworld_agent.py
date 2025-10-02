import asyncio
import os

from transformers import AutoTokenizer

from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.appworld.appworld_env import AppWorldEnv
from rllm.utils import compute_pass_at_k, save_trajectories


async def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 4

    model_name = "gpt-4o-mini"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}
    agent_args = {}
    env_args = {"max_truns": 10}

    # Create engine
    engine = AgentExecutionEngine(
        agent_class=AppWorldReactAgent,
        agent_args=agent_args,
        env_class=AppWorldEnv,
        env_args=env_args,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={"base_url": "https://api.openai.com/v1", "api_key": os.getenv("OPENAI_API_KEY")},
        n_parallel_agents=n_parallel_agents,
        max_response_length=16384,
        max_prompt_length=4096,
        max_steps=10,
    )

    tasks = load_appworld_official_tasks()

    print(f"Running evaluation on {len(tasks)} AppWorld tasks...")
    results = asyncio.run(engine.execute_tasks(tasks))
    # Save trajectories
    save_trajectories(results, save_dir="./trajectories/appworld", filename="trajectories.pt")
    print(f"Pass@k: {compute_pass_at_k(results)}")


def load_appworld_official_tasks():
    """
    Load tasks from the official AppWorld tasks.
    """
    try:
        from appworld import AppWorld

        appworld = AppWorld()

        tasks = appworld.get_tasks(split="test", limit=10)
        print(f"Loaded {len(tasks)} official AppWorld tasks")

        for task in tasks[:3]:
            print(f"Task {task['task_id']}: {task['instruction'][:60]}")
    except Exception as e:
        print(f"Error loading AppWorld tasks: {e}")
        tasks = []

    return tasks


if __name__ == "__main__":
    asyncio.run(main())
