"""Prepare DeepMath dataset for on-policy distillation training.

DeepMath-103K contains math problems for training reasoning models.
This script loads the dataset and registers it for use with rllm trainers.

Usage:
    python -m examples.solver_judge_distill.prepare_deepmath_data
"""

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_deepmath_data():
    """Load and prepare DeepMath dataset for OPD training."""
    train_dataset = load_dataset("zwhe99/DeepMath-103K", split="train")

    test_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    def preprocess_train(example, idx):
        """Convert DeepMath format to solver-judge expected format."""
        return {
            "idx": idx,
            "question": example["question"], 
            "ground_truth": str(example.get("answer", "")), 
            "data_source": "deepmath",
        }

    def preprocess_test(example, idx):
        """Convert AIME format to solver-judge expected format."""
        return {
            "idx": idx,
            "question": example["problem"],
            "ground_truth": str(example["answer"]),
            "data_source": "aime2024",
        }
    # Apply preprocessing
    train_dataset = train_dataset.map(
        preprocess_train, with_indices=True, remove_columns=train_dataset.column_names

    )
    test_dataset = test_dataset.map(
        preprocess_test, with_indices=True, remove_columns=test_dataset.column_names
    )

    # Register datasets under a new name for DeepMath OPD
    train_dataset = DatasetRegistry.register_dataset("deepmath_opd", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("deepmath_opd", test_dataset, "test")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepmath_data()
    print("Train dataset path:", train_dataset.get_data_path())
    print("Test dataset path:", test_dataset.get_data_path())

    # Print samples
    print("\nSample train example:")
    print(train_dataset[0])
    print("\nSample test example:")
    print(test_dataset[0])
