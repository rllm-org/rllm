import importlib
import random

import browsergym.miniwob
import gymnasium as gym

from rllm.data.dataset import DatasetRegistry

importlib.reload(browsergym.miniwob)


def prepare_miniwob_data(train_size=96, test_size=29):
    """
    Prepare and register MiniWoB datasets for training and testing.

    Args:
        train_size (int): Number of training examples to generate
        test_size (int): Number of test examples to generate

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    env_ids = [env_id for env_id in gym.envs.registry.keys() if env_id.startswith("browsergym/miniwob")]
    assert train_size + test_size == len(env_ids), f"Number of training {train_size} and number of testing {test_size} should sum up to total {len(env_ids)}"

    random.seed(42)
    random.shuffle(env_ids)
    # Create train and test data
    train_data = [
        {
            "env_id": env_id,
        }
        for env_id in env_ids[:train_size]
    ]
    test_data = [
        {
            "env_id": env_id,
        }
        for env_id in env_ids[train_size:]
    ]

    # Register the datasets with the DatasetRegistry
    train_dataset = DatasetRegistry.register_dataset("miniwob", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("miniwob", test_data, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_miniwob_data()
    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    print("Sample train example:", train_dataset.get_data()[0])
    print("Sample test example:", test_dataset.get_data()[0])
