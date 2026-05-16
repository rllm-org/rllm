"""Run the Geo3KWorkflow against a live OpenAI-compatible VLM endpoint.

Pre-reqs:
    1. Pull the dataset:    rllm dataset pull geo3k
    2. Start a local server (vllm/sglang) serving a VLM on ``base_url``
       below (defaults to ``http://localhost:30000/v1``).

Usage (from this cookbook directory):
    python run.py
"""

import asyncio
import json
import os
from copy import deepcopy

from geo3k_flow import Geo3KWorkflow
from transformers import AutoProcessor, AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.reward_fn import math_reward_fn


def load_data(n=1):
    """Load geo3k data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("geo3k", "test")
    if dataset is None:
        raise RuntimeError("geo3k test split not found. Run: rllm dataset pull geo3k")

    data = []
    for _, example in enumerate(dataset):
        for _ in range(n):
            data.append(deepcopy(example))
    return data


def evaluate_results(results):
    """Aggregate pass@1 / pass@k from the episode-level correctness flags."""
    from collections import defaultdict

    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    for episode in results:
        idx = episode.task.get("idx") if isinstance(episode.task, dict) else None
        if idx is None:
            idx = episode.id
        problem_correct_map[idx] += int(episode.is_correct)
        problem_total_map[idx] += 1

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

    n_parallel_tasks = 32

    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=4096,
        max_response_length=2048,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    engine = AgentWorkflowEngine(
        workflow_cls=Geo3KWorkflow,
        workflow_args={"reward_function": math_reward_fn},
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    tasks = load_data(n=1)
    print(f"Loaded {len(tasks)} geo3k tasks")

    results = asyncio.run(engine.execute_tasks(tasks))

    print("Evaluating results...")
    evaluate_results(results)

    os.makedirs("logs", exist_ok=True)
    with open("logs/geo3k.json", "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)

    print("\nResults saved to logs/geo3k.json")
