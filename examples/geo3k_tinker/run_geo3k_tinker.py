import asyncio
import json
import os
import sys
from copy import deepcopy

import tinker
from transformers import AutoProcessor, AutoTokenizer

# Import geo3k-specific modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "geo3k"))

from geo3k_workflow import Geo3KWorkflow

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.rewards.reward_fn import math_reward_fn


def load_data(n: int = 1):
    """Load geo3k data using the Dataset interface.

    This mirrors the behavior of the existing OpenAI/vLLM-based runner.
    """
    dataset = DatasetRegistry.load_dataset("geo3k", "test")
    if dataset is None:
        print("Dataset not found, preparing dataset...")
        from preprocess_geo3k import prepare_geo3k_data

        _, dataset = prepare_geo3k_data()

    data = []
    for idx, example in enumerate(dataset):
        for _ in range(n):
            data.append(deepcopy(example))
    return data


def evaluate_results(results):
    """Evaluate the results and compute pass@k metrics."""
    from collections import defaultdict

    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    for episode in results:
        idx = episode.task["idx"]
        is_correct = episode.is_correct
        problem_correct_map[idx] += int(is_correct)
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

    n_parallel_tasks = 128
    n_repeats = 4

    model_name = os.getenv("GEO3K_TINKER_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    base_url = os.getenv("TINKER_BASE_URL")

    service_client = tinker.ServiceClient(base_url=base_url)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load processor for vision-language models
    processor = None
    image_processor = None
    model_name_lower = model_name.lower()
    if "vl" in model_name_lower or "vision" in model_name_lower:
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            if hasattr(processor, "image_processor") and processor.image_processor is not None:
                image_processor = processor.image_processor
        except Exception:
            # If processor loading fails, continue without it
            pass

    rollout_engine = TinkerEngine(
        base_url=base_url,
        model_name=model_name,
        tokenizer=tokenizer,
        service_client=service_client,
        max_prompt_length=1024,
        max_response_length=2048,
        sampling_params={"temperature": 1.0, "top_p": 1.0},
        image_processor=image_processor,
        processor=processor,  # Full processor for VLM support
    )

    # Create a training client and sampling client for inference.
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=4,
    )
    sampler_future = training_client.save_weights_for_sampler(name="000000")
    sampler_result = sampler_future.result()
    sampling_client = training_client.create_sampling_client(sampler_result.path)

    rollout_engine.set_sampling_client(sampling_client)

    engine = AgentWorkflowEngine(
        workflow_cls=Geo3KWorkflow,
        workflow_args={
            "reward_function": math_reward_fn,
            "encode_as_base64": False,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    tasks = load_data(n=n_repeats)
    print(f"Loaded {len(tasks)} geo3k tasks")

    results = asyncio.run(engine.execute_tasks(tasks))

    print("Evaluating results...")
    evaluate_results(results)

    os.makedirs("logs", exist_ok=True)
    output_path = "logs/geo3k_tinker.json"
    with open(output_path, "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)

    print(f"\nResults saved to {output_path}")
