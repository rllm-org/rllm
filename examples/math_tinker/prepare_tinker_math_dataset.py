from datasets import concatenate_datasets, load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_math_data_hendrycks():
    # List of all configs to aggregate
    configs = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

    # Load and collect all splits
    datasets = []
    for config in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
        datasets.append(ds)

    # Aggregate all splits into one dataset
    all_train_dataset = concatenate_datasets(datasets)

    # Optionally, preprocess if needed (example: rename fields, add source, etc.)
    def preprocess_fn(example):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": example.get("problem", ""),
                }
            ],
            "ground_truth": example.get("solution", ""),
            "data_source": "hendrycks_math",
        }

    all_train_dataset = all_train_dataset.map(preprocess_fn)

    train_dataset_fixed = []

    for data in all_train_dataset:
        data["question"] = data["problem"]
        del data["problem"]
        train_dataset_fixed.append(data)

    math_500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_500 = math_500.map(preprocess_fn)

    math_500_fixed = []
    for data in math_500:
        data["question"] = data["problem"]
        del data["problem"]
        math_500_fixed.append(data)

    gsm8k = load_dataset("openai/gsm8k", "main")["train"]
    DatasetRegistry.register_dataset("gsm8k", gsm8k, "train")

    DatasetRegistry.register_dataset("hendrycks_math", train_dataset_fixed, "train")
    DatasetRegistry.register_dataset("math500", math_500_fixed, "test")


if __name__ == "__main__":
    prepare_math_data_hendrycks()
