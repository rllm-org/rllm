import argparse
import os

import numpy as np
import pandas as pd

import rllm
from verl.utils.hdfs_io import copy, makedirs

# Get the directory for rLLM repo (rllm.__file__)
RLLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(rllm.__file__)))


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--local_dir", default=os.path.join(RLLM_DIR, "data/rllm-frozenlake"))
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_size", type=int, default=10000, help="Number of trajectories to generate (default: 10000).")
    parser.add_argument("--test_size", type=int, default=100, help="Number of trajectories to generate (default: 100).")

    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    hdfs_dir = args.hdfs_dir

    np.random.seed(42)
    train_seeds = np.random.randint(0, 100000, size=args.train_size)
    test_seeds = np.random.randint(0, 100000, size=args.test_size)
    train_sizes = np.random.randint(2, 10, size=args.train_size)
    test_sizes = np.random.randint(2, 10, size=args.test_size)
    train_ps = np.random.uniform(0.6, 0.85, size=args.train_size)
    test_ps = np.random.uniform(0.6, 0.85, size=args.test_size)

    def make_map_fn(split):
        def process_fn(seed, size, p, idx):
            return {
                "data_source": "frozenlake",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "bfs",
                "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
                "extra_info": {"split": split, "index": idx, "seed": seed, "size": size, "p": p},
                "uid": f"{seed}_{size}_{p}",
            }

        return process_fn

    train_data = [make_map_fn("train")(seed, train_sizes[idx], train_ps[idx], idx) for idx, seed in enumerate(train_seeds)]
    test_data = [make_map_fn("test")(seed, test_sizes[idx], test_ps[idx], idx) for idx, seed in enumerate(test_seeds)]

    # Convert to DataFrame and save as Parquet
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Copy to HDFS if needed
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


if __name__ == "__main__":
    main()
