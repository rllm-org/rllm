"""Generate and register the FrozenLake train/test datasets.

Each row is a small dict ``{seed, size, p, is_slippery, max_steps, uid}``.
The actual map is regenerated deterministically from these parameters
inside the AgentFlow, so the dataset itself stays tiny.

Usage::

    python cookbooks/frozenlake/prepare_data.py
    python cookbooks/frozenlake/prepare_data.py --train-size 5000 --test-size 200
"""

from __future__ import annotations

import argparse

import numpy as np

from rllm.data.dataset import DatasetRegistry


def _row(seed: int, size: int, p: float, is_slippery: bool, max_steps: int, idx: int) -> dict:
    return {
        "uid": f"frozenlake_{idx:06d}_{seed}_{size}",
        "seed": int(seed),
        "size": int(size),
        "p": float(p),
        "is_slippery": bool(is_slippery),
        "max_steps": int(max_steps),
        "index": int(idx),
        # Stored for human inspection / `rllm dataset list` previews.
        "question": f"FrozenLake puzzle (size={size}, p={p:.2f}, seed={seed})",
    }


def prepare_frozenlake_data(
    train_size: int = 10000,
    test_size: int = 100,
    is_slippery: bool = False,
    rng_seed: int = 42,
):
    rng = np.random.default_rng(rng_seed)

    def _split(n: int, offset: int) -> list[dict]:
        seeds = rng.integers(0, 1_000_000, size=n)
        sizes = rng.integers(2, 10, size=n)
        ps = rng.uniform(0.6, 0.85, size=n)
        return [_row(int(seeds[i]), int(sizes[i]), float(ps[i]), is_slippery, max_steps=2 * int(sizes[i]), idx=offset + i) for i in range(n)]

    train_rows = _split(train_size, offset=0)
    test_rows = _split(test_size, offset=train_size)

    train_dataset = DatasetRegistry.register_dataset("frozenlake", train_rows, "train")
    test_dataset = DatasetRegistry.register_dataset("frozenlake", test_rows, "test")
    return train_dataset, test_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=10000)
    ap.add_argument("--test-size", type=int, default=100)
    ap.add_argument("--slippery", action="store_true")
    args = ap.parse_args()

    train, test = prepare_frozenlake_data(
        train_size=args.train_size,
        test_size=args.test_size,
        is_slippery=args.slippery,
    )
    print(f"Train: {len(train.get_data())} rows")
    print(f"Test:  {len(test.get_data())} rows")
    print("Sample:", train.get_data()[0])


if __name__ == "__main__":
    main()
