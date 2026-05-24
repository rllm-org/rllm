"""Generate and register Sokoban train/test datasets.

Rows contain deterministic generation parameters matching the LaMer Sokoban
main experiment configuration. The actual room is regenerated inside
``sokoban_flow`` from each row's seed and environment settings.
"""

from __future__ import annotations

import argparse

import numpy as np

from rllm.data.dataset import DatasetRegistry


LAMER_SOKOBAN_CONFIG = {
    "train_size": 256,
    "test_size": 64,
    "val_batch_size": 16,
    "env_seed": 4608,
    "dim_room": (6, 6),
    "num_boxes": 2,
    "max_steps": 30,
    "search_depth": 100,
    "min_steps": 5,
    "max_sol_steps": 21,
    "actions_per_turn": 3,
    "max_turns": 7,
    "mode": "text_with_row_numbers",
}


def _row(
    idx: int,
    seed: int,
    split: str,
    dim_room: tuple[int, int],
    num_boxes: int,
    max_steps: int,
    search_depth: int,
    min_steps: int,
    max_sol_steps: int,
    actions_per_turn: int,
    max_turns: int,
    env_seed: int,
) -> dict:
    h, w = dim_room
    return {
        "uid": f"sokoban_{idx:06d}_{seed}_{h}x{w}_{num_boxes}",
        "seed": int(seed),
        "split": split,
        "dim_room": [int(h), int(w)],
        "num_boxes": int(num_boxes),
        "max_steps": int(max_steps),
        "search_depth": int(search_depth),
        "min_steps": int(min_steps),
        "max_sol_steps": int(max_sol_steps),
        "actions_per_turn": int(actions_per_turn),
        "max_turns": int(max_turns),
        "mode": LAMER_SOKOBAN_CONFIG["mode"],
        "lamer_env_seed": int(env_seed),
        "lamer_config_source": "LaMer/examples/sokoban/*_sokoban_qwen3_4b.sh and LaMer/scripts/prepare_example_data.sh",
        "index": int(idx),
        "question": f"Sokoban puzzle ({h}x{w}, boxes={num_boxes}, seed={seed})",
    }


def prepare_sokoban_data(
    train_size: int = LAMER_SOKOBAN_CONFIG["train_size"],
    test_size: int = LAMER_SOKOBAN_CONFIG["test_size"],
    dim_room: tuple[int, int] = LAMER_SOKOBAN_CONFIG["dim_room"],
    num_boxes: int = LAMER_SOKOBAN_CONFIG["num_boxes"],
    max_steps: int = LAMER_SOKOBAN_CONFIG["max_steps"],
    search_depth: int = LAMER_SOKOBAN_CONFIG["search_depth"],
    min_steps: int = LAMER_SOKOBAN_CONFIG["min_steps"],
    max_sol_steps: int | None = LAMER_SOKOBAN_CONFIG["max_sol_steps"],
    actions_per_turn: int = LAMER_SOKOBAN_CONFIG["actions_per_turn"],
    max_turns: int = LAMER_SOKOBAN_CONFIG["max_turns"],
    env_seed: int = LAMER_SOKOBAN_CONFIG["env_seed"],
):
    max_sol_steps = max_steps if max_sol_steps is None else max_sol_steps

    def _split(n: int, offset: int, split: str, rng: np.random.RandomState, low: int, high: int) -> list[dict]:
        seeds = rng.randint(low, high, size=n)
        return [
            _row(
                idx=offset + i,
                seed=int(seeds[i]),
                split=split,
                dim_room=dim_room,
                num_boxes=num_boxes,
                max_steps=max_steps,
                search_depth=search_depth,
                min_steps=min_steps,
                max_sol_steps=max_sol_steps,
                actions_per_turn=actions_per_turn,
                max_turns=max_turns,
                env_seed=env_seed if split == "train" else env_seed + 1000,
            )
            for i in range(n)
        ]

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

    train_dataset = DatasetRegistry.register_dataset("sokoban", train_rows, "train")
    test_dataset = DatasetRegistry.register_dataset("sokoban", test_rows, "test")
    return train_dataset, test_dataset


def _parse_dim(text: str) -> tuple[int, int]:
    parts = [p for p in text.replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return (size, size)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise argparse.ArgumentTypeError("dim-room must be like 6 or 6x6")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Sokoban datasets for rLLM")
    ap.add_argument("--train-size", type=int, default=LAMER_SOKOBAN_CONFIG["train_size"])
    ap.add_argument("--test-size", type=int, default=LAMER_SOKOBAN_CONFIG["test_size"])
    ap.add_argument("--dim-room", type=_parse_dim, default=LAMER_SOKOBAN_CONFIG["dim_room"])
    ap.add_argument("--num-boxes", type=int, default=LAMER_SOKOBAN_CONFIG["num_boxes"])
    ap.add_argument("--max-steps", type=int, default=LAMER_SOKOBAN_CONFIG["max_steps"])
    ap.add_argument("--search-depth", type=int, default=LAMER_SOKOBAN_CONFIG["search_depth"])
    ap.add_argument("--min-steps", type=int, default=LAMER_SOKOBAN_CONFIG["min_steps"])
    ap.add_argument("--max-sol-steps", type=int, default=LAMER_SOKOBAN_CONFIG["max_sol_steps"])
    ap.add_argument("--actions-per-turn", type=int, default=LAMER_SOKOBAN_CONFIG["actions_per_turn"])
    ap.add_argument("--max-turns", type=int, default=LAMER_SOKOBAN_CONFIG["max_turns"])
    ap.add_argument("--env-seed", type=int, default=LAMER_SOKOBAN_CONFIG["env_seed"])
    args = ap.parse_args()

    train, test = prepare_sokoban_data(
        train_size=args.train_size,
        test_size=args.test_size,
        dim_room=args.dim_room,
        num_boxes=args.num_boxes,
        max_steps=args.max_steps,
        search_depth=args.search_depth,
        min_steps=args.min_steps,
        max_sol_steps=args.max_sol_steps,
        actions_per_turn=args.actions_per_turn,
        max_turns=args.max_turns,
        env_seed=args.env_seed,
    )
    print(f"Train: {len(train.get_data())} rows")
    print(f"Test:  {len(test.get_data())} rows")
    print("Sample:", train.get_data()[0])


if __name__ == "__main__":
    main()
