from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_hotpotqa_data(train_size=None, test_size=None):
    """
    Loading HotpotQA dataset and registering it with the DatasetRegistry.
    Only loads essential fields: question, ground_truth, data_source

    Args:
        train_size: Maximum number of training examples to load
        test_size: Maximum number of test examples to load

    Returns:
        tuple: (train_dataset, test_dataset)
    """

    def process_split(split_data, max_size):
        """Process a data split with optional size limit"""
        if max_size is not None:
            split_data = split_data.select(range(min(max_size, len(split_data))))
        print(split_data)
        processed = [{"question": example["question"], "ground_truth": example["answer"], "data_source": "hotpotqa"} for example in split_data]

        print(f"Processed {len(processed)} examples")
        return processed

    print("Loading HotpotQA dataset...")
    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)

    train_processed = process_split(hotpot_dataset["train"], train_size)
    test_processed = process_split(hotpot_dataset["validation"], test_size)

    train_dataset = DatasetRegistry.register_dataset("hotpotqa", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("hotpotqa", test_processed, "test")

    return train_dataset.get_data(), test_dataset.get_data()


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_hotpotqa_data()
    print(f"Train dataset: {train_dataset[0]}")
    print(f"Test dataset: {test_dataset[0]}")
