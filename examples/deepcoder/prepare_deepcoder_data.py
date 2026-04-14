import json

from datasets import concatenate_datasets, load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.data.utils import fetch_live_code_bench_system_prompt


def prepare_deepcoder_data(train_size: int = None, test_size: int = None):
    train_dataset = concatenate_datasets(
        [
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train"),
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train"),
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train"),
        ]
    )
    test_dataset = concatenate_datasets(
        [
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test"),
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test"),
        ]
    )

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

        tests_raw = example["tests"]
        # Handle different test formats
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get("metadata", {})

        # Convert TACO format to standard format
        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
                normalized_tests.append({"input": input_val, "output": output_val, "testtype": "stdin_stdout"})
            tests = normalized_tests

        # Ensure tests is always a list
        if not isinstance(tests, list):
            tests = [tests] if tests else []

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}

        return {
            "question": question,
            "ground_truth": json.dumps(tests),
            "data_source": "livecodebench",
            "uid": f"deepcoder_{idx}",
            "index": idx,
            "starter_code": starter_code,
            "metadata": json.dumps(metadata),
        }

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    train_dataset = DatasetRegistry.register_dataset("deepcoder", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("deepcoder", test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepcoder_data()
    print(f"  - Train dataset: {len(train_dataset.get_data())} examples")
    print(f"  - Test dataset: {len(test_dataset.get_data())} examples")
    print(train_dataset.get_data()[0])
