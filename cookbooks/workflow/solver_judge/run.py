"""Run the SolverJudgeWorkflow against a live OpenAI-compatible endpoint.

Pre-reqs:
    1. Prepare the countdown dataset:
        python examples/countdown/prepare_countdown_data.py
    2. Start a local server (e.g. sglang/vllm) serving the model on
       ``base_url`` below — defaults to ``http://localhost:30000/v1``.

Usage (from this cookbook directory):
    python run_solver_judge_flow.py
"""

import asyncio
import json
import os
import sys
from copy import deepcopy

from solver_judge_flow import SolverJudgeWorkflow
from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.countdown_reward import countdown_reward_fn

# Reuse countdown's prepare script if the dataset isn't registered yet.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "countdown"))


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
        for _ in range(n):
            data.append(deepcopy(processed))
    return data


def process_countdown_fn(example, idx):
    """Process countdown example into the expected task format."""
    question = example["question"]
    target = example["target"]
    nums = example["nums"]

    ground_truth = {"target": target, "numbers": nums}
    return {
        "question": question,
        "ground_truth": ground_truth,
        "idx": idx,
        "data_source": "countdown",
        "target": target,
        "nums": nums,
    }


def evaluate_results(results):
    """Aggregate pass@1 / pass@k from the episode-level correctness flags."""
    from collections import defaultdict

    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    for episode in results:
        problem = episode.task["question"]
        is_correct = episode.is_correct
        problem_correct_map[problem] += int(is_correct)
        problem_total_map[problem] += 1

    k = max(problem_total_map.values()) if problem_total_map else 1
    total_problems = len(problem_correct_map)

    if total_problems > 0:
        pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
        pass_at_k = sum(1 for _, correct in problem_correct_map.items() if correct > 0) / total_problems
    else:
        pass_at_1 = 0.0
        pass_at_k = 0.0

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print(f"Average Pass@{k} Accuracy:", pass_at_k)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_tasks = 128
    n_solutions = 2

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
        workflow_cls=SolverJudgeWorkflow,
        workflow_args={
            "n_solutions": n_solutions,
            "reward_function": countdown_reward_fn,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    tasks = load_data(n=1)
    print(f"Loaded {len(tasks)} countdown tasks")

    results = asyncio.run(engine.execute_tasks(tasks))

    print("Evaluating results...")
    evaluate_results(results)

    os.makedirs("logs", exist_ok=True)
    with open("logs/solver_judge_countdown.json", "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)

    print("\nResults saved to logs/solver_judge_countdown.json")
