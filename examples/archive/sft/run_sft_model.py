import argparse
import asyncio

from transformers import AutoTokenizer

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser(description="Evaluate trained SFT math model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (e.g., outputs/qwen2.5_math_sft/global_step_2784/)")
    args = parser.parse_args()

    n_parallel_agents = 64

    model_name = args.model_path

    # model_name = "Qwen/Qwen2.5-Math-7B-Instruct" # base model eval

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    agent_args = {
        "tools": ["python"],
        "parser_name": "qwen",
        "system_prompt": ('You are an expert mathematician and programmer. Your goal is to solve challenging math problems, like those from the AIME competition, by breaking them down into logical steps and using Python code for calculations. Strive for clarity and efficiency.\n\nFollow this process for every problem:\n1.  **Analyze the Problem**: Read the question carefully. Identify the key information, constraints, and what is being asked.\n2.  **Think Step-by-Step**: In the `<think>` block, outline your plan. Decompose the problem into the smallest, most logical steps. **You must not write code or perform calculations in this block.** Your goal is to create a plan that will be executed by the Python tool.\n3.  **Write Python Code**: In the `<tool_call>` block, write an efficient Python script to execute your plan. The tool expects a JSON object with `name` and `arguments` keys. The `arguments` should be a dictionary with a single `code` key. Ensure the code is self-contained, runs quickly, and prints the final result.\n4.  **State the Final Answer**: After receiving the `<tool_result>`, verify it. Then, state the final answer clearly and concisely in the format \\boxed{answer}.\n\nHere is an example:\nQuestion: What is the largest prime factor of 25! ?\n<think>The problem asks for the largest prime factor of 25 factorial. The largest prime factor of n! is the largest prime number less than or equal to n. In this case, n=25. I will write a Python script to find the largest prime number less than or equal to 25.</think>\n<tool_call>\n{"name": "python", "arguments": {"code": "import math\\ndef is_prime(n):\\n    if n <= 1:\\n        return False\\n    for i in range(2, int(math.sqrt(n)) + 1):\\n        if n % i == 0:\\n            return False\\n    return True\\n\\ndef largest_prime_up_to(n):\\n    for i in range(n, 1, -1):\\n        if is_prime(i):\\n            return i\\n    return None\\n\\nprint(largest_prime_up_to(25))"}}\n</tool_call>\n<tool_result>\n23\n</tool_result>\nThe largest prime factor of 25! is the largest prime number less than or equal to 25. The answer is \\boxed{23}.'),
    }
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=ToolAgent,
        agent_args=agent_args,
        env_class=ToolEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=14000,
        max_prompt_length=2000,
        n_parallel_agents=n_parallel_agents,
        max_steps=20,  # Allow more steps for complex problems
    )

    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_math_data import prepare_math_data

        _, test_dataset = prepare_math_data()

    tasks = test_dataset.repeat(n=8)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
