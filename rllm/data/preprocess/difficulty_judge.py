import argparse
import concurrent.futures
import json
import os
from concurrent.futures import as_completed
from copy import deepcopy
from tqdm import tqdm

# Replace these imports with your actual module paths
from rllm.system_prompts import MATH_DIFFICULTY_PROMPT
from rllm.utils import call_gemini_llm

# Import the dataset loading logic
from rllm.data.dataset_types import (
    TrainDataset,
    TestDataset,
)
from rllm.data.utils import load_dataset

def difficulty_fn(idx, entry):
    """
    1) Extract problem and solution text.
    2) Call LLM for difficulty estimates (8 numeric strings).
    3) Convert to float safely, filter out parse errors.
    4) Take the average and store as 'difficulty'.
    """
    # if entry.get('difficulty') is not None:
    #     # Skip if already computed
    #     return idx, entry

    problem_text = entry.get('problem', '')
    solution_text = entry.get('solution', '')

    # Pass@8 difficulty calls
    output_list = call_gemini_llm(
        f"Problem: {problem_text}\n----\nSolution: {solution_text}",
        system_prompt=MATH_DIFFICULTY_PROMPT,
        n=8,
        temperature=0.7,
    )
    # (Use .lower() to catch both uppercase/lowercase errors)
    output_list = [
        o for o in output_list
        if 'error' not in o.lower()
    ]

    # Attempt to parse each string as float
    values = []
    for o in output_list:
        try:
            val = float(o)
            values.append(val)
        except ValueError:
            # Ignore anything that can't be parsed as float
            pass

    # Compute the average or set None if no valid floats
    if values:
        difficulty = sum(values) / len(values)
    else:
        difficulty = None
        print('Failed parsing all difficulties: ', output_list)

    entry['difficulty'] = difficulty
    return idx, entry


def batch_difficulty(dataset: str, split: str):

    # Figure out if we need a TrainDataset or TestDataset
    if split == "train":
        dataset_enum = TrainDataset[dataset.upper()]
    else:
        dataset_enum = TestDataset[dataset.upper()]

    # Load data using the provided load_dataset function
    data = load_dataset(dataset_enum)
    results = deepcopy(data)

    # Prepare to save back to the same file location
    data_dir = "train" if isinstance(dataset_enum, TrainDataset) else "test"
    dataset_name = dataset_enum.value.lower()
    file_path = os.path.join("..", data_dir, f"{dataset_name}.json")

    # Use ThreadPoolExecutor to process concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(difficulty_fn, i, entry)
            for i, entry in enumerate(data)
        ]
        done_count = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, result = future.result()
            results[idx] = result
            done_count += 1

            # Periodically save partial results
            if done_count % 5000 == 0:
                print(f"Processed {done_count} entries so far. Saving partial results...")
                with open(file_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    # Save final results
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Finished processing {len(results)} entries. Results saved to {file_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label approximate difficulty for math problems.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Name of the dataset (e.g. 'AIME', 'AMC', 'OMNI_MATH', 'OLYMPIAD', 'MATH')"
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "test"],
        help="Which split to use: 'train' or 'test'"
    )
    args = parser.parse_args()
    batch_difficulty(args.dataset, args.split)
