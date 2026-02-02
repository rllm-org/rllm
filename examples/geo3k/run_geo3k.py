import asyncio
import json
import os
from copy import deepcopy

from geo3k_workflow import Geo3KWorkflow

from rllm.data.dataset import DatasetRegistry
from rllm.engine import AgentWorkflowEngine, OpenAIEngine
from rllm.rewards.reward_fn import math_reward_fn


def load_data(n=1):
    """Load geo3k data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("geo3k", "test")
    if dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_geo3k_data import preprocess_geo3k_data

        _, dataset = preprocess_geo3k_data()

    data = []
    for idx, example in enumerate(dataset):
        for i in range(n):
            data.append(deepcopy(example))
    return data


def evaluate_results(results):
    """Evaluate the results and compute pass@k metrics."""
    from collections import defaultdict

    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    # Count correct answers for each problem
    for episode in results:
        idx = episode.task["idx"]

        # Use the episode-level is_correct flag set by the workflow
        is_correct = episode.is_correct

        problem_correct_map[idx] += int(is_correct)
        problem_total_map[idx] += 1

    # Calculate pass@1 and pass@k
    k = max(problem_total_map.values()) if problem_total_map else 1
    total_problems = len(problem_correct_map)

    if total_problems > 0:
        pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
        pass_at_k = sum(1 for idx, correct in problem_correct_map.items() if correct > 0) / total_problems
    else:
        pass_at_1 = 0.0
        pass_at_k = 0.0

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print(f"Average Pass@{k} Accuracy:", pass_at_k)


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_tasks = 128
    model_name = "Qwen/Qwen3-VL-2B-Instruct"

    rollout_engine = OpenAIEngine(
        model=model_name,
        max_prompt_length=1024,
        max_response_length=2048,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    engine = AgentWorkflowEngine(
        workflow_cls=Geo3KWorkflow,
        workflow_args={
            "reward_function": math_reward_fn,
            "encode_as_base64": True,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    tasks = load_data(n=4)
    print(f"Loaded {len(tasks)} geo3k tasks")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Evaluate results (rewards are already assigned in the workflow)
    print("Evaluating results...")
    evaluate_results(results)

    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/geo3k.json", "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)

    print("\nResults saved to logs/geo3k.json")
