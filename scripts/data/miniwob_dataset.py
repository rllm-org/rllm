import argparse
import os
import random

import browsergym.miniwob
import gymnasium as gym
import pandas as pd

import rllm
from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    import importlib
    import os

    import browsergym.miniwob

    importlib.reload(browsergym.miniwob)
    # Get the directory for rLLM repo (rllm.__file__)
    RLLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(rllm.__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.path.join(RLLM_DIR, "data/rllm-miniwob"))
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.768, help="Ratio of data to use for training (default: 76.8%)")
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    hdfs_dir = args.hdfs_dir
    train_ratio = max(0.0, min(1.0, args.train_ratio))

    # Get all MiniWoB environment IDs from gym
    env_ids = [env_id for env_id in gym.envs.registry.keys() if env_id.startswith("browsergym/miniwob")]
    random.seed(42)
    random.shuffle(env_ids)

    def make_map_fn(split):
        def process_fn(env_id, idx):
            data = {
                "data_source": "miniwob",
                "prompt": [
                    {
                        "role": "user",
                        "content": "",  # placeholder since there is no real prompt is needed to environment based trajectory collection
                    }
                ],
                "ability": "web",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "env_id": env_id,  # field for env based data
                },
            }

            return data

        return process_fn

    # Split train/test
    train_size = int(train_ratio * len(env_ids))  # 80% for training
    train_envs = env_ids[:train_size]
    test_envs = env_ids[train_size:]

    # Process train data
    train_data = [make_map_fn("train")(env_id, idx) for idx, env_id in enumerate(train_envs)]

    # Process test data
    test_data = [make_map_fn("test")(env_id, idx) for idx, env_id in enumerate(test_envs)]

    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))

    # Convert to DataFrame and save as Parquet
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Copy to HDFS if needed
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
