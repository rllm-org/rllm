"""Prepare GSM8K dataset for AgentCore math agent training,
link to example: https://github.com/awslabs/agentcore-rl-toolkit/tree/main/examples/strands_math_agent

The strands math agent expects {"prompt": ..., "answer": ...} fields,
so we transform GSM8K's native format to match.

Usage:
    python -m examples.agentcore_math.prepare_gsm8k_data
"""

import re

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


# Adapted from verl/examples/data_preprocess/gsm8k.py
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def prepare_gsm8k_data():
    """Load GSM8K and register with field names matching the ACR math agent."""
    hf_train = load_dataset("openai/gsm8k", "main", split="train")
    hf_test = load_dataset("openai/gsm8k", "main", split="test")

    def transform(example, idx):
        return {
            "idx": idx,
            "prompt": example["question"],
            "answer": extract_solution(example["answer"]),
            "data_source": "gsm8k",
        }

    hf_train = hf_train.map(transform, with_indices=True, remove_columns=hf_train.column_names)
    hf_test = hf_test.map(transform, with_indices=True, remove_columns=hf_test.column_names)

    train_ds = DatasetRegistry.register_dataset(
        "gsm8k_agentcore",
        hf_train,
        "train",
        source="openai/gsm8k",
        description="GSM8K with prompt/answer fields for AgentCore math agent",
        category="math",
    )
    test_ds = DatasetRegistry.register_dataset(
        "gsm8k_agentcore",
        hf_test,
        "test",
        source="openai/gsm8k",
        description="GSM8K with prompt/answer fields for AgentCore math agent",
        category="math",
    )

    print(f"Train: {len(train_ds)} examples")
    print(f"Test:  {len(test_ds)} examples")
    print(f"Sample: {train_ds[0]}")
    return train_ds, test_ds


if __name__ == "__main__":
    prepare_gsm8k_data()
