from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_geo3k_data():
    # Load dataset
    dataset = load_dataset("hiyouga/geometry3k")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # instruction_following = (
    #     r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    #     r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    #     r"The final answer MUST BE put in \boxed{}."
    # )

    instruction_following = "Let's think step by step and output your final answer in \\boxed{}."

    def process_fn(example, idx):
        problem = example.pop("problem")
        prompt = problem + instruction_following
        answer = example.pop("answer")
        image = example.pop("images")

        data = {
            "idx": idx,
            "data_source": "geo3k",
            "image": image,
            "question": prompt,
            "ground_truth": answer,
        }
        return data

    # Preprocess datasets
    train_dataset = train_dataset.map(function=process_fn, with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=process_fn, with_indices=True, num_proc=8)

    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("geo3k", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("geo3k", test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_geo3k_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())
