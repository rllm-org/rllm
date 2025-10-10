"""
Run Frozen Lake Workflow with rllm

This script demonstrates how to execute frozen lake tasks using rllm's
AgentWorkflowEngine with eval-protocol's MCPGymRolloutProcessor.
"""

import asyncio
import json
import os
from pathlib import Path

import eval_protocol
from frozen_lake_flow import FrozenLakeWorkflow

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.openai_engine import OpenAIEngine


def load_frozen_lake_data(max_tasks: int = 3):
    """
    Load frozen lake dataset from eval_protocol package.

    Args:
        max_tasks: Maximum number of tasks to load

    Returns:
        List of task dictionaries with structure:
        {
            "id": "frozen_lake_task_0",
            "system_prompt": "...",
            "environment_context": {...},
            "user_prompt_template": "{observation}"
        }
    """
    # Load dataset from local file
    dataset_path = Path(__file__).parent / "frozen_lake_dataset.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Frozen Lake dataset not found at {dataset_path}. "
            "Please ensure eval-protocol is properly installed."
        )

    tasks = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_tasks:
                break
            task = json.loads(line)
            tasks.append(task)

    print(f"Loaded {len(tasks)} frozen lake tasks")
    return tasks


def evaluate_results(episodes):
    """
    Evaluate the results and compute accuracy metrics.

    Args:
        episodes: List of Episode objects
    """
    total = len(episodes)
    correct = sum(1 for ep in episodes if ep.is_correct)
    accuracy = correct / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total tasks: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print()

    for episode in episodes:
        status = "✅" if episode.is_correct else "❌"
        reward = episode.metrics.get("frozen_lake_reward", 0.0)
        print(f"{status} Task {episode.id}: reward={reward:.3f}")

    print("=" * 60)

    return accuracy


async def main():
    """Main execution function."""

    n_parallel_tasks = 3
    max_tasks = 3
    model_id = "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"

    # Create dummy rollout_engine (required by Workflow base class but not used)
    rollout_engine = OpenAIEngine(
        model="unused",
        base_url="https://api.openai.com/v1",
        api_key="unused",
    )

    engine = AgentWorkflowEngine(
        workflow_cls=FrozenLakeWorkflow,
        workflow_args={
            "steps": 30,
            "model": model_id,
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    tasks = load_frozen_lake_data(max_tasks=max_tasks)

    print(f"Starting frozen lake workflow execution...")
    print(f"Model: {model_id}")
    print(f"Parallel tasks: {n_parallel_tasks}")
    print()

    try:
        episodes = await engine.execute_tasks(tasks)

        accuracy = evaluate_results(episodes)

        output_dir = Path("logs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "frozen_lake_results.json"

        with open(output_file, "w") as f:
            json.dump([episode.to_dict() for episode in episodes], f, indent=2)

        print(f"\n✅ Results saved to {output_file}")

        return accuracy

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        engine.shutdown()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    accuracy = asyncio.run(main())

    print(f"\n🎯 Final Accuracy: {accuracy:.2%}")