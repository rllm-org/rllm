import asyncio
import json
import os

from transformers import AutoTokenizer

from examples.solver_judge_distill.solver_judge_math_workflow import (
    SolverJudgeMathWorkflow,
    math_reward_fn,
)
from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine


def load_data(n_samples: int = 1):
    """Load AIME test data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("solver_judge_math", "test")
    if dataset is None:
        print("Dataset not found. Please run prepare_math_data.py first:")
        print("  python -m examples.solver_judge_distill.prepare_math_data")
        return None

    data = []
    for idx, example in enumerate(dataset):
        task = {
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "data_source": example.get("data_source", "math"),
            "idx": idx,
        }
        # Optionally repeat each task n_samples times
        for _ in range(n_samples):
            data.append(task.copy())
    return data


def evaluate_results(results):
    """Evaluate results and compute metrics."""
    from collections import defaultdict

    problem_results = defaultdict(list)

    for episode in results:
        problem = episode.task["question"]
        problem_results[problem].append(episode.is_correct)

    # Calculate metrics
    total_problems = len(problem_results)
    total_correct = sum(1 for p, results in problem_results.items() if any(results))

    # Pass@1: Average accuracy across all attempts
    all_attempts = [r for results in problem_results.values() for r in results]
    pass_at_1 = sum(all_attempts) / len(all_attempts) if all_attempts else 0.0

    # Pass@k: Whether at least one attempt per problem was correct
    pass_at_k = total_correct / total_problems if total_problems > 0 else 0.0

    # Get solver and judge metrics
    solver_accs = []
    judge_accs = []
    for episode in results:
        if episode.metrics:
            solver_accs.append(episode.metrics.get("solver_acc", 0))
            judge_accs.append(episode.metrics.get("judge_acc", 0))

    avg_solver_acc = sum(solver_accs) / len(solver_accs) if solver_accs else 0.0
    avg_judge_acc = sum(judge_accs) / len(judge_accs) if judge_accs else 0.0

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total unique problems: {total_problems}")
    print(f"Pass@1 Accuracy: {pass_at_1:.2%}")
    print(f"Pass@k Accuracy: {pass_at_k:.2%}")
    print(f"Average Solver Accuracy: {avg_solver_acc:.2%}")
    print(f"Average Judge Accuracy: {avg_judge_acc:.2%}")
    print("=" * 50)

    return {
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "solver_acc": avg_solver_acc,
        "judge_acc": avg_judge_acc,
        "total_problems": total_problems,
    }


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_tasks = 32
    model_name = "Qwen/Qwen3-8B"
    base_url = "http://localhost:30000/v1"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Connecting to model server at {base_url}...")
    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        max_prompt_length=2048,
        max_response_length=2048,
        base_url=base_url,
        api_key="EMPTY",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    engine = AgentWorkflowEngine(
        workflow_cls=SolverJudgeMathWorkflow,
        workflow_args={
            "n_solutions": 2,
            "reward_function": math_reward_fn,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    # Load test data
    tasks = load_data(n_samples=1)
    if tasks is None:
        exit(1)

    print(f"Loaded {len(tasks)} test tasks from AIME 2024")
    print("Running evaluation...")

    # Execute evaluation
    results = asyncio.run(engine.execute_tasks(tasks))

    # Evaluate and print results
    metrics = evaluate_results(results)

    # Save results
    os.makedirs("logs", exist_ok=True)
    output_path = "logs/solver_judge_math_baseline.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "episodes": [episode.to_dict() for episode in results],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_path}")

