import asyncio
import json
import os
import sys
from copy import deepcopy

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.workflows.simple_workflow import SimpleWorkflow

# Add countdown directory to path for prepare_countdown_data import
sys.path.append(os.path.dirname(__file__))


def load_data(n=1):
    """Load countdown data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("countdown", "test")
    if dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_countdown_data import prepare_countdown_data

        _, dataset, _, _ = prepare_countdown_data()

    data = []
    for idx, example in enumerate(dataset):
        processed = process_countdown_fn(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_countdown_fn(example, idx):
    """Process countdown example into the expected format."""
    question = example["question"]
    target = example["target"]
    nums = example["nums"]

    # Create ground truth in the format expected by countdown_reward_fn
    ground_truth = {"target": target, "numbers": nums}

    task = {"question": question, "ground_truth": ground_truth, "idx": idx, "data_source": "countdown", "target": target, "nums": nums}
    return task


def evaluate_results(results):
    """Evaluate the results and compute pass@k metrics."""
    from collections import defaultdict

    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    # Count correct answers for each problem
    for episode in results:
        problem = episode.task["question"]

        # Use the episode-level is_correct flag set by the workflow
        is_correct = episode.is_correct

        problem_correct_map[problem] += int(is_correct)
        problem_total_map[problem] += 1

    # Calculate pass@1 and pass@k
    k = max(problem_total_map.values()) if problem_total_map else 1
    total_problems = len(problem_correct_map)

    if total_problems > 0:
        pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
        pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems
    else:
        pass_at_1 = 0.0
        pass_at_k = 0.0

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print(f"Average Pass@{k} Accuracy:", pass_at_k)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_tasks = 128

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        max_prompt_length=2048,
        max_response_length=1024,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    engine = AgentWorkflowEngine(
        workflow_cls=SimpleWorkflow,
        workflow_args={
            "reward_function": countdown_reward_fn,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    # Load countdown tasks
    tasks = load_data(n=1)
    print(f"Loaded {len(tasks)} countdown tasks")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Evaluate results (rewards are already assigned in the workflow)
    print("Evaluating results...")
    evaluate_results(results)

    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/countdown.json", "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)

    print("\nResults saved to logs/countdown.json")
