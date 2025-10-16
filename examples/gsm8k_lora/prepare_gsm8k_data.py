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
    gsm8k_dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = gsm8k_dataset["train"]
    test_dataset = gsm8k_dataset["test"]

    def preprocess_fn(example, idx):
        return {
            "question": example["question"],
            "ground_truth": extract_solution(example["answer"]),
            "data_source": "gsm8k",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    train_dataset = DatasetRegistry.register_dataset("gsm8k", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("gsm8k", test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_gsm8k_data()
    print(train_dataset)
    print(test_dataset)
