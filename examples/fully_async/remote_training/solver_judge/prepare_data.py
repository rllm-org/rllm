"""Prepare the Countdown dataset for remote training.

Downloads from HuggingFace, preprocesses into the question/answer format,
and saves train/test splits locally as JSON files so the client-side
AgentTrainerClient can load them without needing the DatasetRegistry.

Usage:
    python -m examples.fully_async.remote_training.solver_judge.prepare_data \
        --output-dir ./countdown_data
"""

import argparse
import json
import os
import random


def prepare_countdown_data(output_dir: str):
    from datasets import load_dataset

    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    test_size = 1024
    total_size = len(dataset)

    test_dataset = dataset.select(range(test_size))
    train_dataset = dataset.select(range(test_size, total_size))

    def preprocess(example):
        target = example["target"]
        nums = example["nums"]
        nums_str = ", ".join(map(str, nums))
        question = (
            f"Using the numbers {nums_str}, find a way to reach the target "
            f"number {target}. You can use basic arithmetic operations "
            f"(+, -, *, /) and each number can only be used once. Show your "
            f"step-by-step calculation and output the final answer within "
            f"<answer>...</answer>, for example <answer> (1 + 2) / 3 </answer>."
        )
        return {
            "question": question,
            "ground_truth": str(target),
            "target": target,
            "nums": nums,
        }

    train_examples = [preprocess(ex) for ex in train_dataset]
    test_examples = [preprocess(ex) for ex in test_dataset]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_examples, f, indent=2)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_examples, f, indent=2)

    print(f"Saved {len(train_examples)} train examples to {output_dir}/train.json")
    print(f"Saved {len(test_examples)} test examples to {output_dir}/test.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./countdown_data")
    args = parser.parse_args()
    prepare_countdown_data(args.output_dir)
