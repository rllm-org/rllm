"""Remote Solver-Judge training client.

This script runs on your **local desktop** and connects to a remote
TrainingServer running on a GPU cluster. It implements the Solver-Judge
workflow from ``examples/solver_judge/`` using the remote training
framework.

Usage (after starting the server):

    # 1. Prepare data (only once)
    python -m examples.fully_async.remote_training.solver_judge.prepare_data \
        --output-dir ./countdown_data

    # 2. Run training
    python -m examples.fully_async.remote_training.solver_judge.train \
        --server-url http://<GPU_SERVER_IP>:8000 \
        --model-name Qwen/Qwen3-4B-Instruct-2507 \
        --data-dir ./countdown_data
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial

from torch.utils.data import Dataset

from examples.fully_async.remote_training.solver_judge.solver_judge import rollout_fn
from rllm.experimental.fully_async.remote import AgentTrainerClient


# ── Simple dataset class ────────────────────────────────────────────────────


class CountdownDataset(Dataset):
    """Loads a JSON file produced by ``prepare_data.py``."""

    def __init__(self, path: str):
        with open(path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Remote Solver-Judge training client",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="URL of the remote TrainingServer (e.g. http://gpu-server:8000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model name for local tokenizer loading",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./countdown_data",
        help="Directory containing train.json / test.json from prepare_data.py",
    )
    parser.add_argument("--n-solutions", type=int, default=2, help="Solver candidates per problem")
    parser.add_argument("--n", type=int, default=4, help="GRPO group size (rollouts per prompt)")
    parser.add_argument("--max-concurrency", type=int, default=128)
    args = parser.parse_args()

    # ── Load dataset ──
    train_path = os.path.join(args.data_dir, "train.json")
    if not os.path.exists(train_path):
        print(
            f"ERROR: {train_path} not found. Run prepare_data.py first:\n"
            f"  python -m examples.fully_async.remote_training.solver_judge.prepare_data "
            f"--output-dir {args.data_dir}"
        )
        return

    dataset = CountdownDataset(train_path)
    print(f"Loaded {len(dataset)} training examples from {train_path}")

    # ── Wrap rollout_fn with n_solutions ──
    # partial() bakes in the n_solutions kwarg so the signature stays
    # async def fn(client, tokenizer, **kwargs) -> Trajectory
    solver_judge_rollout = partial(rollout_fn, n_solutions=args.n_solutions)

    # ── Create client trainer ──
    trainer = AgentTrainerClient(
        server_url=args.server_url,
        rollout_fn=solver_judge_rollout,
        model_name=args.model_name,
        dataset=dataset,
        n=args.n,
        max_concurrency=args.max_concurrency,
        config_overrides={
            # ── Algorithm ──
            "algorithm.adv_estimator": "grpo",

            # ── Async training ──
            "async_training.required_samples": 64,
            "async_training.trigger_parameter_sync_step": 4,

            # ── Rollout ──
            "rollout.n": args.n,
            "rollout.total_rollout_steps": len(dataset),
            "rollout.test_freq": 10,

            # ── Actor ──
            "actor_rollout_ref.actor.optim.lr": 1e-6,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
            "actor_rollout_ref.actor.clip_ratio_low": 0.2,
            "actor_rollout_ref.actor.clip_ratio_high": 0.28,

            # ── Logging ──
            "trainer.project_name": "remote-solver-judge",
            "trainer.experiment_name": "countdown-solver-judge",
            "trainer.logger": ["console", "wandb"],
            "trainer.save_freq": 100,
            "trainer.total_epochs": 10,
        },
    )

    trainer.train()


if __name__ == "__main__":
    main()
