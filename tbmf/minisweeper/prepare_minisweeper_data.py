"""Generate and register MiniSweeper train/test datasets.

Rows contain pre-generated puzzle states matching the LaMer Minesweeper main
experiment configuration, so training/eval do not regenerate boards at reset.
"""

from __future__ import annotations

import argparse

import numpy as np

from env_service.minesweeper import MineSweeperEnv
from rllm.data.dataset import DatasetRegistry


LAMER_MINISWEEPER_CONFIG = {
    "train_size": 256,
    "test_size": 64,
    "val_batch_size": 16,
    "env_seed": 0,
    "board_size": 6,
    "n_mines": 3,
    "board_type": "board",
    "mode": "text",
    "max_steps": 15,
    "max_turns": 7,
}


def _row(
    idx: int,
    seed: int,
    split: str,
    board_size: int,
    n_mines: int,
    max_steps: int,
    max_turns: int,
    env_seed: int,
    puzzle_state: dict,
) -> dict:
    return {
        "uid": f"minisweeper_{idx:06d}_{seed}_{board_size}_{n_mines}",
        "seed": int(seed),
        "split": split,
        "board_size": int(board_size),
        "n_mines": int(n_mines),
        "board_type": LAMER_MINISWEEPER_CONFIG["board_type"],
        "mode": LAMER_MINISWEEPER_CONFIG["mode"],
        "max_steps": int(max_steps),
        "max_turns": int(max_turns),
        "lamer_env_seed": int(env_seed),
        "lamer_config_source": "LaMer/examples/minesweeper/*_minesweeper_qwen3_4b.sh and LaMer/scripts/prepare_example_data.sh",
        "puzzle_state": puzzle_state,
        "index": int(idx),
        "question": f"MiniSweeper puzzle (board={board_size}x{board_size}, mines={n_mines}, seed={seed})",
    }


def _generate_puzzle_state(*, seed: int, board_size: int, n_mines: int, board_type: str) -> dict:
    env = MineSweeperEnv(board_size=board_size, n_mines=n_mines, board_type=board_type, seed=seed)
    try:
        env.reset()
        return env.export_puzzle_state()
    finally:
        env.close()


def prepare_minisweeper_data(
    train_size: int = LAMER_MINISWEEPER_CONFIG["train_size"],
    test_size: int = LAMER_MINISWEEPER_CONFIG["test_size"],
    board_size: int = LAMER_MINISWEEPER_CONFIG["board_size"],
    n_mines: int = LAMER_MINISWEEPER_CONFIG["n_mines"],
    max_steps: int = LAMER_MINISWEEPER_CONFIG["max_steps"],
    max_turns: int = LAMER_MINISWEEPER_CONFIG["max_turns"],
    env_seed: int = LAMER_MINISWEEPER_CONFIG["env_seed"],
):
    if n_mines >= board_size * board_size:
        raise ValueError("n_mines must be smaller than board_size * board_size")

    def _split(n: int, offset: int, split: str, rng: np.random.RandomState, low: int, high: int) -> list[dict]:
        seeds = rng.randint(low, high, size=n)
        rows = []
        for i in range(n):
            seed = int(seeds[i])
            puzzle_state = _generate_puzzle_state(
                seed=seed,
                board_size=board_size,
                n_mines=n_mines,
                board_type=LAMER_MINISWEEPER_CONFIG["board_type"],
            )
            rows.append(
                _row(
                    idx=offset + i,
                    seed=seed,
                    split=split,
                    board_size=board_size,
                    n_mines=n_mines,
                    max_steps=max_steps,
                    max_turns=max_turns,
                    env_seed=env_seed if split == "train" else env_seed + 1000,
                    puzzle_state=puzzle_state,
                )
            )
        return rows

    train_rows = _split(
        train_size,
        offset=0,
        split="train",
        rng=np.random.RandomState(env_seed),
        low=0,
        high=2**16 - 1,
    )
    test_rows = _split(
        test_size,
        offset=train_size,
        split="test",
        rng=np.random.RandomState(env_seed + 1000),
        low=2**16,
        high=2**32 - 1,
    )

    train_dataset = DatasetRegistry.register_dataset("minisweeper", train_rows, "train")
    test_dataset = DatasetRegistry.register_dataset("minisweeper", test_rows, "test")
    return train_dataset, test_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare MiniSweeper datasets for rLLM")
    ap.add_argument("--train-size", type=int, default=LAMER_MINISWEEPER_CONFIG["train_size"])
    ap.add_argument("--test-size", type=int, default=LAMER_MINISWEEPER_CONFIG["test_size"])
    ap.add_argument("--board-size", type=int, default=LAMER_MINISWEEPER_CONFIG["board_size"])
    ap.add_argument("--n-mines", type=int, default=LAMER_MINISWEEPER_CONFIG["n_mines"])
    ap.add_argument("--max-steps", type=int, default=LAMER_MINISWEEPER_CONFIG["max_steps"])
    ap.add_argument("--max-turns", type=int, default=LAMER_MINISWEEPER_CONFIG["max_turns"])
    ap.add_argument("--env-seed", type=int, default=LAMER_MINISWEEPER_CONFIG["env_seed"])
    args = ap.parse_args()

    train, test = prepare_minisweeper_data(
        train_size=args.train_size,
        test_size=args.test_size,
        board_size=args.board_size,
        n_mines=args.n_mines,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        env_seed=args.env_seed,
    )
    print(f"Train: {len(train.get_data())} rows")
    print(f"Test:  {len(test.get_data())} rows")
    print("Sample:", train.get_data()[0])


if __name__ == "__main__":
    main()
