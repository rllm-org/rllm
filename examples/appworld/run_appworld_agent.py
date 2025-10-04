import argparse
import asyncio
import os

from transformers import AutoTokenizer

from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.appworld.appworld_env import AppWorldEnv

# ============================================================================
# Fix AppWorld multithreading issues: apply signal patch
# ============================================================================
from rllm.environments.appworld.signal_patch import apply_signal_patch
from rllm.utils import compute_pass_at_k, save_trajectories

# Apply patch before importing AppWorld
# Signal can only be used in the main thread but in the async engine, the thread is not the main thread.
apply_signal_patch(verbose=True)
# ============================================================================


async def main(num_tasks=10, max_turns=40, split="dev"):
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
    env_args = {"max_turns": max_turns}

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
        max_steps=max_turns,
    )

    tasks = load_appworld_official_tasks(split=split, num_tasks=num_tasks)

    if not tasks:
        print("No tasks loaded, exiting...")
        return

    print(f"Running evaluation on {len(tasks)} AppWorld tasks...")
    results = await engine.execute_tasks(tasks)

    # Save trajectories
    save_trajectories(results, save_dir="./trajectories/appworld", filename="trajectories.pt")
    compute_pass_at_k(results)
    # Compute accuracy and show per-task results
    print("\n" + "=" * 80)
    print("Task Completion Results")
    print("=" * 80)
    n_passed = 0
    for i, trajectory in enumerate(results, 1):
        task_id = trajectory.task.get("task_id", f"task_{i}") if isinstance(trajectory.task, dict) else f"task_{i}"
        reward = trajectory.reward
        status = "PASSED" if reward >= 1.0 else "FAILED"

        print(f"{i:2d}. {task_id:20s} | Reward: {reward:.2f} | {status}")

        if reward >= 1.0:
            n_passed += 1

    accuracy = n_passed / num_tasks if num_tasks > 0 else 0.0

    print("=" * 80)
    print(f"Summary: {n_passed} out of {num_tasks} tasks passed")
    print(f"Accuracy: {accuracy:.2%} ({n_passed}/{num_tasks})")
    print("=" * 80 + "\n")


def load_appworld_official_tasks(split="dev", num_tasks=10):
    """
    Load tasks from the official AppWorld tasks.
    """
    try:
        # lazy load the appworld package
        from appworld import AppWorld, load_task_ids

        # Use 'dev' split for development/testing
        # Available splits: 'train', 'dev', 'test_normal', 'test_challenge'
        task_ids = load_task_ids(split)[:num_tasks]  # Get first 10 task IDs

        # Create task dictionaries with task_id
        # The AppWorldEnv will load the instruction when it initializes
        tasks = []
        for task_id in task_ids:
            # Temporarily create AppWorld instance to get instruction for display
            try:
                world = AppWorld(task_id=task_id)
                instruction = world.task.instruction
            except Exception:
                instruction = f"Task {task_id}"

            tasks.append({"task_id": task_id, "instruction": instruction})

        print(f"Loaded {len(tasks)} official AppWorld tasks from 'dev' split")

        for task in tasks:
            print(f"Task {task['task_id']}: {task['instruction'][:80]}...")
        return tasks
    except Exception as e:
        print(f"Warning: Cannot load AppWorld - {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AppWorld Agent with rLLM", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num-tasks", type=int, default=10, help="Number of tasks to run (use -1 for all tasks)")
    parser.add_argument("-t", "--max-turns", type=int, default=40, help="Maximum number of turns per task")
    parser.add_argument("-s", "--split", type=str, default="dev", choices=["train", "dev", "test_normal", "test_challenge"], help="Which split to use")

    args = parser.parse_args()

    asyncio.run(main(num_tasks=args.num_tasks, max_turns=args.max_turns, split=args.split))
