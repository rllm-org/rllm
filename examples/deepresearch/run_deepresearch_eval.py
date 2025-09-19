"""
DeepResearch Evaluation Script using rLLM AgentWorkflowEngine

This script runs DeepResearch evaluation on various datasets using the integrated
rLLM workflow engine. It demonstrates how to use the DeepResearch agent within
the rLLM framework for research tasks.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any

from deepresearch_tools import get_all_tools
from deepresearch_workflow import DeepResearchWorkflow
from dotenv import find_dotenv, load_dotenv
from transformers import AutoTokenizer

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import OpenAIEngine


def load_sample_tasks(max_samples: int = 5) -> list[dict[str, Any]]:
    """
    Load sample research tasks for testing.

    Args:
        max_samples: Maximum number of samples to generate

    Returns:
        List of task dictionaries
    """
    # Sample research questions for testing
    sample_questions = [
        {
            "question": "What is the capital of France and what is its population?",
            "answer": "Paris, approximately 2.16 million",
            "task_type": "factual",
        },
        {
            "question": "Calculate the area of a circle with radius 5 units.",
            "answer": "78.54 square units",
            "task_type": "mathematical",
        },
        {
            "question": "What are the main causes of climate change?",
            "answer": "Greenhouse gas emissions, deforestation, industrial processes",
            "task_type": "analytical",
        },
        {
            "question": "Who won the Nobel Prize in Physics in 2023?",
            "answer": "Pierre Agostini, Ferenc Krausz, and Anne L'Huillier",
            "task_type": "factual",
        },
        {
            "question": "Explain the difference between machine learning and deep learning.",
            "answer": "Machine learning is broader, deep learning uses neural networks with multiple layers",
            "task_type": "conceptual",
        },
    ]

    tasks = []
    for i, sample in enumerate(sample_questions[:max_samples]):
        task = {
            "id": f"sample_{i}",
            "question": sample["question"],
            "answer": sample["answer"],
            "task_type": sample["task_type"],
            "metadata": {
                "source": "sample_data",
                "difficulty": "medium",
                "timestamp": datetime.now().isoformat(),
            },
        }
        tasks.append(task)

    return tasks


def load_gaia_tasks(dataset_path: str, max_samples: int = None) -> list[dict[str, Any]]:
    """
    Load tasks from GAIA dataset.

    Args:
        dataset_path: Path to GAIA dataset file
        max_samples: Maximum number of samples to load

    Returns:
        List of task dictionaries
    """
    if not os.path.exists(dataset_path):
        print(f"GAIA dataset not found at {dataset_path}")
        print("Using sample tasks instead...")
        return load_sample_tasks(max_samples or 5)

    try:
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        tasks = []
        items = data if isinstance(data, list) else [data]

        for i, item in enumerate(items):
            if max_samples and i >= max_samples:
                break

            task = {
                "id": f"gaia_{i}",
                "question": item.get("question", item.get("query", "")),
                "answer": item.get("answer", ""),
                "task_type": "gaia",
                "metadata": {
                    "source": "gaia",
                    "level": item.get("level", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                },
            }
            tasks.append(task)

        print(f"Loaded {len(tasks)} tasks from GAIA dataset")
        return tasks

    except Exception as e:
        print(f"Error loading GAIA dataset: {e}")
        print("Using sample tasks instead...")
        return load_sample_tasks(max_samples or 5)


def setup_rollout_engine(args) -> OpenAIEngine:
    """
    Set up the OpenAI rollout engine.

    Args:
        args: Command line arguments

    Returns:
        Configured OpenAI engine
    """
    # Load environment variables
    load_dotenv(find_dotenv())

    # Provider selection (similar to Strands)
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Allow command line override
    if args.api_key:
        api_key = args.api_key
        base_url = args.base_url or "https://api.openai.com/v1"
        model_name = args.model or "gpt-4"
    elif together_api_key:
        api_key = together_api_key
        base_url = args.base_url or "https://api.together.xyz/v1"
        model_name = args.model or os.getenv(
            "TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo"
        )
        print("🔧 Using Together AI API")
    elif openai_api_key:
        api_key = openai_api_key
        base_url = args.base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        model_name = args.model or os.getenv("MODEL_NAME", "gpt-4")
        print("🔧 Using OpenAI API")
    else:
        raise ValueError(
            "❌ API key required. Please set OPENAI_API_KEY or TOGETHER_AI_API_KEY in .env file"
        )

    # Set up tokenizer if available
    tokenizer = None
    if args.tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"✅ Loaded tokenizer: {args.tokenizer}")
        except Exception as e:
            print(f"⚠️  Could not load tokenizer {args.tokenizer}: {e}")
            tokenizer = None

    # Create OpenAI engine
    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
        api_key=api_key,
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
    )

    print("✅ Created OpenAI engine:")
    print(f"   Model: {model_name}")
    print(f"   Base URL: {base_url}")
    print(f"   Temperature: {args.temperature}")

    return rollout_engine


def save_results(results: list[Any], output_path: str):
    """
    Save evaluation results to file.

    Args:
        results: List of episode results
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert episodes to serializable format
    serializable_results = []
    for episode in results:
        episode_dict = {
            "id": episode.id,
            "task": episode.task,
            "is_correct": episode.is_correct,
            "termination_reason": episode.termination_reason.value
            if episode.termination_reason
            else None,
            "metrics": episode.metrics,
            "trajectories": [],
        }

        # Add trajectory information
        for agent_name, trajectory in episode.trajectories:
            trajectory_dict = {
                "agent_name": agent_name,
                "task": trajectory.task,
                "reward": trajectory.reward,
                "num_steps": len(trajectory.steps),
                "steps": [],
            }

            # Add step information (simplified)
            for step in trajectory.steps:
                step_dict = {
                    "model_response": step.model_response[
                        :500
                    ],  # Truncate for readability
                    "action": step.action.__dict__ if step.action else None,
                    "observation": step.observation[:200] if step.observation else "",
                    "reward": step.reward,
                }
                trajectory_dict["steps"].append(step_dict)

            episode_dict["trajectories"].append(trajectory_dict)

        serializable_results.append(episode_dict)

    # Save to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {output_path}")


def print_evaluation_summary(results: list[Any]):
    """
    Print a summary of evaluation results.

    Args:
        results: List of episode results
    """
    total_tasks = len(results)
    correct_tasks = sum(1 for episode in results if episode.is_correct)
    accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0.0

    # Count termination reasons
    termination_counts = {}
    for episode in results:
        reason = (
            episode.termination_reason.value
            if episode.termination_reason
            else "unknown"
        )
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    # Calculate average metrics
    total_rounds = sum(episode.metrics.get("rounds", 0) for episode in results)
    total_time = sum(episode.metrics.get("time_taken", 0) for episode in results)
    avg_rounds = total_rounds / total_tasks if total_tasks > 0 else 0
    avg_time = total_time / total_tasks if total_tasks > 0 else 0

    print("\n" + "=" * 60)
    print("📊 DEEPRESEARCH EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {total_tasks}")
    print(f"Correct answers: {correct_tasks}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average rounds per task: {avg_rounds:.1f}")
    print(f"Average time per task: {avg_time:.1f}s")
    print("\nTermination reasons:")
    for reason, count in termination_counts.items():
        print(f"  {reason}: {count}")
    print("=" * 60)


async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Run DeepResearch evaluation using rLLM"
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        choices=["sample", "gaia"],
        default="sample",
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--gaia-path",
        default="../../../../rllm/data/train/web/gaia.json",
        help="Path to GAIA dataset file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Maximum number of samples to evaluate",
    )

    # Model options
    parser.add_argument("--model", default="gpt-4", help="Model name to use")
    parser.add_argument(
        "--base-url", default="https://api.openai.com/v1", help="API base URL"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (uses OPENAI_API_KEY env var if not provided)",
    )
    parser.add_argument(
        "--tokenizer", default=None, help="Tokenizer model name (optional)"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens per response"
    )

    # Execution options
    parser.add_argument(
        "--parallel-tasks", type=int, default=4, help="Number of parallel tasks"
    )
    parser.add_argument(
        "--output-dir", default="./outputs", help="Output directory for results"
    )

    args = parser.parse_args()

    print("🚀 Starting DeepResearch Evaluation")
    print("=" * 50)

    # Load tasks
    if args.dataset == "gaia":
        tasks = load_gaia_tasks(args.gaia_path, args.max_samples)
    else:
        tasks = load_sample_tasks(args.max_samples)

    print(f"📋 Loaded {len(tasks)} tasks")

    # Set up rollout engine
    rollout_engine = setup_rollout_engine(args)

    # Get tools
    tools = get_all_tools()
    print(f"🔧 Loaded {len(tools)} tools: {list(tools.keys())}")

    # Create workflow engine
    engine = AgentWorkflowEngine(
        workflow_cls=DeepResearchWorkflow,
        workflow_args={
            "tools": tools,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=args.parallel_tasks,
        retry_limit=1,
    )

    print(f"⚙️  Created AgentWorkflowEngine with {args.parallel_tasks} parallel tasks")

    # Run evaluation
    print("\n🔬 Starting evaluation...")
    start_time = asyncio.get_event_loop().time()

    try:
        results = await engine.execute_tasks(tasks)
        end_time = asyncio.get_event_loop().time()

        print(f"\n✅ Evaluation completed in {end_time - start_time:.1f}s")

        # Print summary
        print_evaluation_summary(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.output_dir, f"deepresearch_eval_{timestamp}.json"
        )
        save_results(results, output_path)

        # Print some example results
        print("\n📝 Sample results:")
        for i, episode in enumerate(results[:2]):  # Show first 2 results
            print(
                f"\nTask {i + 1}: {episode.task.get('question', 'No question')[:100]}..."
            )
            print(f"Prediction: {episode.metrics.get('prediction', 'No prediction')}")
            print(f"Correct: {episode.is_correct}")
            print(f"Rounds: {episode.metrics.get('rounds', 0)}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Set environment for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    asyncio.run(main())
