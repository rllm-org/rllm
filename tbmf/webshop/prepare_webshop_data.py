"""Generate and register WebShop train/test datasets.

Each row stores the session id plus the WebShop environment settings used by
the flow. The actual shopping episode is created at runtime by
``webshop_flow`` using the local ``webshop_env.WebShopEnv`` wrapper.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from rllm.data.dataset import DatasetRegistry


def _resolve_webshop_data_root() -> Path:
    env_root = os.environ.get("WEBSHOP_DATA_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"WEBSHOP_DATA_ROOT points to a missing path: {candidate}")

    search_root = Path(__file__).resolve().parent
    for parent in search_root.parents:
        candidate = parent / "datasets" / "webshop" / "webshop_data"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "WebShop data root not found. Set WEBSHOP_DATA_ROOT or create datasets/webshop/webshop_data."
    )


LAMER_WEBSHOP_CONFIG = {
    "train_size": 1000,
    "test_size": 500,
    "env_seed": 42,
    "observation_mode": "text",
    "num_products": 1000,
    "human_goals": False,
    "max_steps": 30,
    "max_turns": 30,
    "use_available_actions": True,
    "use_accumulate_history": True,
    "use_accumulate_thinking": True,
}


def _split_indices(
    n: int,
    rng: np.random.RandomState,
    low: int,
    high: int,
) -> list[int]:
    if n <= 0:
        return []
    population = high - low
    if n > population:
        return list(range(low, high))
    return [int(v) for v in rng.choice(np.arange(low, high), size=n, replace=False)]


def _row(
    idx: int,
    session_id: int,
    split: str,
    *,
    data_root: Path,
    env_seed: int,
    observation_mode: str,
    num_products: int | None,
    human_goals: bool,
    max_steps: int,
    max_turns: int,
    use_available_actions: bool,
    use_accumulate_history: bool,
    use_accumulate_thinking: bool,
) -> dict:
    return {
        "uid": f"webshop_{split}_{session_id}",
        "session_id": int(session_id),
        "split": split,
        "index": int(idx),
        "seed": int(env_seed),
        "observation_mode": observation_mode,
        "num_products": num_products,
        "human_goals": bool(human_goals),
        "max_steps": int(max_steps),
        "max_turns": int(max_turns),
        "use_available_actions": bool(use_available_actions),
        "use_accumulate_history": bool(use_accumulate_history),
        "use_accumulate_thinking": bool(use_accumulate_thinking),
        "file_path": str(data_root / "items_shuffle_1000.json"),
        "attr_path": str(data_root / "items_ins_v2_1000.json"),
        "question": f"WebShop session {session_id}",
        "instruction": f"WebShop session {session_id}",
        "lamer_config_source": "rLLM WebShop dataset split",
    }


def prepare_webshop_data(
    train_size: int = LAMER_WEBSHOP_CONFIG["train_size"],
    test_size: int = LAMER_WEBSHOP_CONFIG["test_size"],
    seed: int = LAMER_WEBSHOP_CONFIG["env_seed"],
    use_full_data: bool = False,
    *,
    data_root: str | Path | None = None,
    env_seed: int = LAMER_WEBSHOP_CONFIG["env_seed"],
    observation_mode: str = LAMER_WEBSHOP_CONFIG["observation_mode"],
    num_products: int | None = LAMER_WEBSHOP_CONFIG["num_products"],
    human_goals: bool = LAMER_WEBSHOP_CONFIG["human_goals"],
    max_steps: int = LAMER_WEBSHOP_CONFIG["max_steps"],
    max_turns: int = LAMER_WEBSHOP_CONFIG["max_turns"],
    use_available_actions: bool = LAMER_WEBSHOP_CONFIG["use_available_actions"],
    use_accumulate_history: bool = LAMER_WEBSHOP_CONFIG["use_accumulate_history"],
    use_accumulate_thinking: bool = LAMER_WEBSHOP_CONFIG["use_accumulate_thinking"],
):
    """Register WebShop train/test datasets."""
    data_root_path = Path(data_root).expanduser() if data_root is not None else _resolve_webshop_data_root()
    if not data_root_path.exists():
        raise FileNotFoundError(
            f"WebShop data root not found: {data_root_path}\n"
            "Expected the webshop_data directory under datasets/webshop/ or set WEBSHOP_DATA_ROOT."
        )

    train_start_idx = 1500
    train_end_idx = 12000
    test_end_idx = 500
    rng = np.random.RandomState(seed)

    if use_full_data:
        train_indices = list(range(train_start_idx, train_end_idx))
        test_indices = list(range(test_end_idx))
    else:
        train_indices = _split_indices(train_size, rng, train_start_idx, train_end_idx)
        test_indices = _split_indices(test_size, np.random.RandomState(seed + 1000), 0, test_end_idx)

    train_rows = [
        _row(
            idx=i,
            session_id=session_id,
            split="train",
            data_root=data_root_path,
            env_seed=env_seed,
            observation_mode=observation_mode,
            num_products=num_products,
            human_goals=human_goals,
            max_steps=max_steps,
            max_turns=max_turns,
            use_available_actions=use_available_actions,
            use_accumulate_history=use_accumulate_history,
            use_accumulate_thinking=use_accumulate_thinking,
        )
        for i, session_id in enumerate(train_indices)
    ]
    test_rows = [
        _row(
            idx=i,
            session_id=session_id,
            split="test",
            data_root=data_root_path,
            env_seed=env_seed,
            observation_mode=observation_mode,
            num_products=num_products,
            human_goals=human_goals,
            max_steps=max_steps,
            max_turns=max_turns,
            use_available_actions=use_available_actions,
            use_accumulate_history=use_accumulate_history,
            use_accumulate_thinking=use_accumulate_thinking,
        )
        for i, session_id in enumerate(test_indices)
    ]

    train_dataset = DatasetRegistry.register_dataset("webshop", train_rows, "train")
    test_dataset = DatasetRegistry.register_dataset("webshop", test_rows, "test")
    return train_dataset, test_dataset


def _parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WebShop datasets for rLLM")
    parser.add_argument("--train-size", type=int, default=LAMER_WEBSHOP_CONFIG["train_size"])
    parser.add_argument("--test-size", type=int, default=LAMER_WEBSHOP_CONFIG["test_size"])
    parser.add_argument("--seed", type=int, default=LAMER_WEBSHOP_CONFIG["env_seed"])
    parser.add_argument("--use-full-data", action="store_true")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--env-seed", type=int, default=LAMER_WEBSHOP_CONFIG["env_seed"])
    parser.add_argument("--observation-mode", type=str, default=LAMER_WEBSHOP_CONFIG["observation_mode"])
    parser.add_argument("--num-products", type=int, default=LAMER_WEBSHOP_CONFIG["num_products"])
    parser.add_argument("--human-goals", type=_parse_bool, default=LAMER_WEBSHOP_CONFIG["human_goals"])
    parser.add_argument("--max-steps", type=int, default=LAMER_WEBSHOP_CONFIG["max_steps"])
    parser.add_argument("--max-turns", type=int, default=LAMER_WEBSHOP_CONFIG["max_turns"])
    parser.add_argument("--use-available-actions", type=_parse_bool, default=LAMER_WEBSHOP_CONFIG["use_available_actions"])
    parser.add_argument("--use-accumulate-history", type=_parse_bool, default=LAMER_WEBSHOP_CONFIG["use_accumulate_history"])
    parser.add_argument("--use-accumulate-thinking", type=_parse_bool, default=LAMER_WEBSHOP_CONFIG["use_accumulate_thinking"])
    args = parser.parse_args()

    train_dataset, test_dataset = prepare_webshop_data(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        use_full_data=args.use_full_data,
        data_root=args.data_root,
        env_seed=args.env_seed,
        observation_mode=args.observation_mode,
        num_products=args.num_products,
        human_goals=args.human_goals,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        use_available_actions=args.use_available_actions,
        use_accumulate_history=args.use_accumulate_history,
        use_accumulate_thinking=args.use_accumulate_thinking,
    )

    print(f"Train: {len(train_dataset.get_data())} rows")
    print(f"Test:  {len(test_dataset.get_data())} rows")
    if train_dataset.get_data():
        print("Sample train row:", train_dataset.get_data()[0])


if __name__ == "__main__":
    main()
