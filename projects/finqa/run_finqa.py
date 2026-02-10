# Standard Imports
import argparse
import asyncio
import hashlib
import json
import os
from collections import defaultdict

from transformers import AutoTokenizer

from projects.finqa.fin_qa_agent import FinQAAgent
from projects.finqa.fin_qa_environment import FinQAEnvironment
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--n", type=int, default=4, help="Attempts per task")
    parser.add_argument("--output", type=str, help="JSON output file")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95}

    engine = AgentExecutionEngine(
        agent_class=FinQAAgent,
        env_class=FinQAEnvironment,
        engine_name="openai",
        rollout_engine_args={
            "model": model_name,
            "base_url": f"http://localhost:{args.port}/v1",
            "api_key": "None",
            "force_chat_completions": True,  # vLLM 0.11.2 completions endpoint is buggy
        },
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        n_parallel_agents=256,
        max_steps=50,
        max_prompt_length=4096,
        max_response_length=12288,
    )

    test_dataset = DatasetRegistry.load_dataset("finqa_benchmark", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from projects.finqa.prepare_finqa_data import prepare_finqa_data

        _, _, test_dataset = prepare_finqa_data()

    tasks = test_dataset.repeat(n=args.n)

    results = asyncio.run(engine.execute_tasks(tasks))

    # Compute metrics
    problem_correct = defaultdict(int)
    for r in results:
        task_str = json.dumps(r.task, sort_keys=True) if isinstance(r.task, dict) else str(r.task)
        task_hash = hashlib.md5(task_str.encode()).hexdigest()
        problem_correct[task_hash] += 1 if r.reward > 0 else 0

    num_tasks = len(problem_correct)
    total_correct = sum(problem_correct.values())
    pass_at_1 = total_correct / len(results) if results else 0.0
    pass_at_k = sum(1 for c in problem_correct.values() if c > 0) / num_tasks if num_tasks else 0.0
    avg_reward = sum(r.reward for r in results) / len(results) if results else 0.0

    print(f"Model: {model_name}")
    print(f"Completed {len(results)} tasks ({num_tasks} unique × {args.n})")
    print(f"Pass@1: {pass_at_1:.2%} | Pass@{args.n}: {pass_at_k:.2%} | Avg Reward: {avg_reward:.4f}")

    if args.output:
        output_data = {
            "model": model_name,
            "n": args.n,
            "pass@1": pass_at_1,
            f"pass@{args.n}": pass_at_k,
            "avg_reward": avg_reward,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
