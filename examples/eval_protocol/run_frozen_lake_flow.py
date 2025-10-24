"""
Run Frozen Lake Workflow with rllm-fw

This script demonstrates how to execute frozen lake tasks using rllm-fw's
AgentWorkflowEngine with eval-protocol's MCPGymRolloutProcessor.
"""

import asyncio
import json
import os
from pathlib import Path

from frozen_lake_flow import FrozenLakeWorkflow

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.openai_engine import OpenAIEngine


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

    n_parallel_tasks = 4
    max_tasks = 4
    model_id = "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507"

    # Create dummy rollout_engine (required by Workflow base class but not used)
    rollout_engine = OpenAIEngine(
        model=model_id,
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.getenv("FIREWORKS_API_KEY"),
    )

    engine = AgentWorkflowEngine(
        workflow_cls=FrozenLakeWorkflow,
        workflow_args={
            "lite_llm_prefix": "fireworks_ai/",
            "steps": 30,
            "temperature": 1.0,
            "max_tokens": 16384,
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    test_dataset = DatasetRegistry.load_dataset("frozen_lake_eval_protocol", "test")
    tasks = []
    for i in range(max_tasks):
        tasks.append(test_dataset[i])

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
