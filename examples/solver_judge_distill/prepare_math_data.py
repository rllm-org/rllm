from datasets import load_dataset, concatenate_datasets

from rllm.data.dataset import DatasetRegistry


def prepare_math_data():
    """
    Prepare math datasets for solver-judge training.

    - Training: DeepScaleR dataset (diverse math problems)
    - Test: AIME 2024 (competition-level problems for evaluation)

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Load datasets from HuggingFace
    train_dataset = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
    test_dataset1 = load_dataset("HuggingFaceH4/aime_2024", split="train")
    test_dataset2 = load_dataset("MathArena/aime_2025", split="train")

    def preprocess_fn(example, idx):
        """Convert dataset format to solver-judge expected format."""
        return {
            "idx": idx,
            "question": example["problem"],
            "ground_truth": str(example["answer"]),
            "data_source": "math",
        }

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset1 = test_dataset1.map(preprocess_fn, with_indices=True, remove_columns=test_dataset1.column_names)
    test_dataset2 = test_dataset2.map(preprocess_fn, with_indices=True, remove_columns=test_dataset2.column_names)
    test_dataset = concatenate_datasets([test_dataset1, test_dataset2])

    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("solver_judge_math", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("solver_judge_math", test_dataset, "test")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()
    print("Train dataset path:", train_dataset.get_data_path())
    print("Test dataset path:", test_dataset.get_data_path())

    # Print samples
    print("\nSample train example:")
    print(train_dataset[0])
    print("\nSample test example:")
    print(test_dataset[0])

