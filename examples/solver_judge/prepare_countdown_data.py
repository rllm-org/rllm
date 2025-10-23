import random

from rllm.data.dataset import DatasetRegistry


def prepare_countdown_data():
    """
    Prepare the countdown task dataset from HuggingFace.
    Take 1024 examples as test set, remaining as training set.
    Also create stage 2 and stage 3 training sets with 50k examples each.
    """
    # Load the countdown dataset
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    # Split dataset: 1024 examples for test, rest for training
    test_size = 1024
    total_size = len(dataset)

    # Create train/test split
    test_dataset = dataset.select(range(test_size))
    train_dataset = dataset.select(range(test_size, total_size))

    def preprocess_fn(example, idx):
        """
        Convert countdown task format to math problem format.
        Example: target=98, nums=[44, 19, 35] becomes a math word problem.
        """
        target = example["target"]
        nums = example["nums"]

        # Format as a math problem
        nums_str = ", ".join(map(str, nums))
        question = f"Using the numbers {nums_str}, find a way to reach the target number {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your step-by-step calculation and output the final answer within <answer>...</answer>, for example <answer> (1 + 2) / 3 </answer>."

        return {
            "question": question,
            "ground_truth": str(target),
            "data_source": "countdown",
            "target": target,
            "nums": nums,
        }

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    # Create stage 2 and stage 3 training datasets
    train_size = len(train_dataset)
    stage_size = 50000

    # Ensure we have enough data for both stages
    if train_size < 2 * stage_size:
        print(f"Warning: Training set has only {train_size} examples, but need {2 * stage_size} for both stages")
        stage_size = min(stage_size, train_size // 2)

    # Shuffle and select indices for stage 2 and stage 3
    all_indices = list(range(train_size))
    random.shuffle(all_indices)

    stage2_indices = all_indices[:stage_size]
    stage3_indices = all_indices[stage_size : 2 * stage_size]

    # Create stage datasets
    stage2_dataset = train_dataset.select(stage2_indices)
    stage3_dataset = train_dataset.select(stage3_indices)

    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("countdown", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("countdown", test_dataset, "test")
    stage2_dataset = DatasetRegistry.register_dataset("countdown", stage2_dataset, "stage2_train")
    stage3_dataset = DatasetRegistry.register_dataset("countdown", stage3_dataset, "stage3_train")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Stage 2 train dataset size: {len(stage2_dataset)}")
    print(f"Stage 3 train dataset size: {len(stage3_dataset)}")

    return train_dataset, test_dataset, stage2_dataset, stage3_dataset


if __name__ == "__main__":
    train_dataset, test_dataset, stage2_dataset, stage3_dataset = prepare_countdown_data()
    print("Train dataset path:", train_dataset.get_data_path())
    print("Test dataset path:", test_dataset.get_data_path())
    print("Stage 2 train dataset path:", stage2_dataset.get_data_path())
    print("Stage 3 train dataset path:", stage3_dataset.get_data_path())

    # Print a sample
    print("\nSample train example:")
    print(train_dataset[0])
    print("\nSample stage 2 train example:")
    print(stage2_dataset[0])
    print("\nSample stage 3 train example:")
    print(stage3_dataset[0])
