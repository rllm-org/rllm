import asyncio
import os

from transformers import AutoTokenizer

from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.appworld.appworld_env import AppWorldEnv
from rllm.utils import compute_pass_at_k, save_trajectories


async def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY")
        return

    n_parallel_agents = 4

    model_name = "gpt-4o-mini"
    # Use a tokenizer with chat template (only for formatting messages and calculating token counts in the engine)
    # Qwen2-0.5B is small and fast to download
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}
    agent_args = {}
    env_args = {"max_turns": 10}

    # Create engine
    engine = AgentExecutionEngine(
        agent_class=AppWorldReactAgent,
        agent_args=agent_args,
        env_class=AppWorldEnv,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={"base_url": "https://api.openai.com/v1", "api_key": os.getenv("OPENAI_API_KEY")},
        n_parallel_agents=n_parallel_agents,
        max_response_length=16384,
        max_prompt_length=4096,
        max_steps=10,
    )

    tasks = load_appworld_official_tasks()

    if not tasks:
        print("No tasks loaded, exiting...")
        return

    print(f"Running evaluation on {len(tasks)} AppWorld tasks...")
    results = await engine.execute_tasks(tasks)

    # Save trajectories
    save_trajectories(results, save_dir="./trajectories/appworld", filename="trajectories.pt")
    compute_pass_at_k(results)


def load_appworld_official_tasks():
    """
    Load tasks from the official AppWorld tasks.
    """
    try:
        from appworld import AppWorld

        appworld = AppWorld()

        tasks = appworld.get_tasks(split="test", limit=2)
        print(f"Loaded {len(tasks)} official AppWorld tasks")

        for task in tasks[:3]:
            print(f"Task {task['task_id']}: {task['instruction'][:60]}")
        return tasks
    except Exception as e:
        print(f"Warning: Cannot load AppWorld - {e}")
        print("Using mock tasks for testing...")

        # Create mock tasks
        tasks = [
            {
                "task_id": "mock_001",
                "instruction": "Find all playlists in the Spotify app and count them.",
            },
            {
                "task_id": "mock_002",
                "instruction": "Check today's calendar events.",
            },
        ]
        print(f"Created {len(tasks)} mock tasks for testing")
        return tasks


if __name__ == "__main__":
    asyncio.run(main())
