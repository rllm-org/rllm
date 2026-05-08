#!/usr/bin/env python3
"""Run N-pass SWE-bench evaluation using the AgentFlow/Evaluator framework.

Evaluates every instance N times to measure pass@k metrics.

Usage:
    python swe/scripts/run_n_eval.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3.5-35B-A3B \
        --dataset swe_smith_py \
        --n_runs 3 --n_parallel 50 \
        --output_dir results/n_eval/

    # Resume from saved trajectories
    python swe/scripts/run_n_eval.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3.5-35B-A3B \
        --dataset swe_smith_py \
        --n_runs 5 --n_parallel 400 \
        --output_dir results/prev_run/ --resume
"""

import argparse
import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from dotenv import load_dotenv

from swe.flow_config import add_flow_cli_args, flow_config_from_args
from swe.prepare_data import prepare_dataset, load_prepared_data, DATASET_CONFIGS

load_dotenv(_PROJECT_ROOT / ".env")

EMPTY_TRAJ = {
    "messages": [],
    "is_correct": False,
    "patch": "",
    "exit_status": "missing",
    "error": "Episode missing from results",
}


def _extract_traj_data(episode) -> dict:
    """Extract per-rollout data from an Episode."""
    return {
        "is_correct": episode.is_correct,
        "patch": episode.artifacts["patch"],
        "exit_status": episode.artifacts["exit_status"],
        "messages": episode.artifacts["messages"],
        "segments": episode.artifacts.get("segments", []),
    }


def _load_saved_trajectories(trajectories_dir: Path, instance_ids: list[str]) -> dict[str, dict[int, dict]]:
    """Load previously saved per-file trajectories from trajectories/{instance_id}/{rollout_idx}.json."""
    saved: dict[str, dict[int, dict]] = {}
    if not trajectories_dir.is_dir():
        return saved
    instance_set = set(instance_ids)
    for inst_dir in trajectories_dir.iterdir():
        if not inst_dir.is_dir() or inst_dir.name not in instance_set:
            continue
        rollouts: dict[int, dict] = {}
        for json_file in inst_dir.glob("*.json"):
            try:
                rollout_idx = int(json_file.stem)
                with json_file.open() as f:
                    rollouts[rollout_idx] = json.load(f)
            except (ValueError, json.JSONDecodeError, OSError):
                continue
        if rollouts:
            saved[inst_dir.name] = rollouts
    return saved


def _save_trajectory(trajectories_dir: Path, instance_id: str, rollout_idx: int, traj_data: dict) -> None:
    """Save a single trajectory to trajectories/{instance_id}/{rollout_idx}.json."""
    inst_dir = trajectories_dir / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)
    with (inst_dir / f"{rollout_idx}.json").open("w") as f:
        json.dump(traj_data, f)


async def main():
    parser = argparse.ArgumentParser(
        description="Run N-pass SWE-bench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Inference
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", required=True)

    # N-run
    parser.add_argument("--n_runs", type=int, default=3)

    # Execution
    parser.add_argument("--n_parallel", type=int, default=50)
    parser.add_argument("--slice", default="")
    parser.add_argument("--instance_ids", nargs="+", default=None)

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved trajectories in output_dir/trajectories/")

    # Output
    parser.add_argument("--output_dir", required=True)

    # SWEAgentFlow config — one helper registers all flat flags.
    add_flow_cli_args(parser)
    # N-eval-appropriate defaults; all flags can still be overridden on the CLI.
    parser.set_defaults(
        cost_limit=50.0,
        step_limit=100,
        command_timeout=120,
        sandbox_timeout=3900,
        agent_timeout=3600,
        sandbox_retry_attempts=2,
    )

    # Misc
    parser.add_argument("--dockerhub_username", default="jefzda")
    parser.add_argument("--scripts_dir", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"

    # ---- Load tasks ----
    if args.data_path:
        tasks = load_prepared_data(args.data_path)
    else:
        tasks = prepare_dataset(args.dataset, args.split, dockerhub_username=args.dockerhub_username)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(tasks)
    if args.instance_ids:
        instance_set = set(args.instance_ids)
        tasks = [t for t in tasks if t["instance_id"] in instance_set]
    if args.slice:
        parts = [int(x) if x else None for x in args.slice.split(":")]
        tasks = tasks[slice(*parts)]

    n_instances = len(tasks)

    # ---- Load saved trajectories for resume ----
    saved_rollouts: dict[str, dict[int, dict]] = {}
    if args.resume:
        saved_rollouts = _load_saved_trajectories(
            trajectories_dir,
            [t["instance_id"] for t in tasks],
        )
    n_saved = sum(len(v) for v in saved_rollouts.values())

    # ---- Expand tasks N times, skipping completed rollouts ----
    expanded_tasks = []
    for task in tasks:
        iid = task["instance_id"]
        completed = saved_rollouts.get(iid, {})
        for rollout_idx in range(args.n_runs):
            if rollout_idx in completed:
                continue
            expanded_tasks.append(task | {"_rollout_idx": rollout_idx})

    print(f"\nN-Run Evaluation: {n_instances} instances x {args.n_runs} runs = {n_instances * args.n_runs} total")
    print(f"  base_url: {args.base_url}")
    print(f"  model:    {args.model}")
    if args.resume:
        print(f"  Resume: {n_saved} saved rollouts found, {len(expanded_tasks)} remaining")

    if not expanded_tasks:
        print("All rollouts already completed. Nothing to do.")
        return 0

    # ---- Run via EvalRunner ----
    from swe.agent_flow import SWEAgentFlow
    from swe.evaluator import SWEEvaluator
    from rllm.eval.runner import run_dataset
    from rllm.types import Task

    agent = SWEAgentFlow(flow_config_from_args(args))
    evaluator = SWEEvaluator(
        scripts_dir=args.scripts_dir,
        dockerhub_username=args.dockerhub_username,
        command_timeout=args.command_timeout,
        sandbox_timeout=args.sandbox_timeout,
        max_grading_workers=args.n_parallel,
        verbose=args.verbose,
    )
    # Save each trajectory incrementally as it completes
    def on_episode_complete(_idx, episode):
        if episode is None:
            return
        iid = episode.task["instance_id"]
        rollout_idx = episode.task.get("_rollout_idx")
        if rollout_idx is None:
            return
        traj_data = _extract_traj_data(episode)
        _save_trajectory(trajectories_dir, iid, int(rollout_idx), traj_data)

    _, episodes = await run_dataset(
        [
            Task(
                id=str(task.get("instance_id", "")),
                instruction=str(task.get("problem_statement", "")),
                metadata=task,
                dataset_dir=Path("."),
            )
            for task in expanded_tasks
        ],
        agent,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.n_parallel,
        agent_name="swe",
        dataset_name=args.dataset,
        on_episode_complete=on_episode_complete,
        evaluator_override=evaluator,
    )

    # ---- Group by instance (merge saved + new) ----
    rollouts_by_instance: dict[str, dict[int, dict]] = defaultdict(dict)

    # Start with saved rollouts
    for iid, rollouts in saved_rollouts.items():
        for rollout_idx, traj_data in rollouts.items():
            if 0 <= rollout_idx < args.n_runs:
                rollouts_by_instance[iid][rollout_idx] = traj_data

    # Add new results
    for episode in episodes:
        if episode is None:
            continue
        iid = episode.task["instance_id"]
        rollout_idx = episode.task.get("_rollout_idx")
        if rollout_idx is None:
            continue
        rollouts_by_instance[iid][int(rollout_idx)] = _extract_traj_data(episode)

    # ---- Assemble results ----
    results = []
    for task in tasks:
        iid = task["instance_id"]
        saved = rollouts_by_instance.get(iid, {})
        entry = {"instance_id": iid, "trajectories": [], "task": task}
        n_correct = 0
        n_completed = 0
        for rollout_idx in range(args.n_runs):
            traj_data = saved.get(rollout_idx, EMPTY_TRAJ)
            if rollout_idx in saved:
                n_completed += 1
                if traj_data.get("is_correct"):
                    n_correct += 1
            entry["trajectories"].append(traj_data)
        entry["success_rate"] = n_correct / args.n_runs
        entry["n_runs_completed"] = n_completed
        results.append(entry)

    # ---- Save ----
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    avg_success = sum(r["success_rate"] for r in results) / len(results) if results else 0.0
    n_any = sum(1 for r in results if r["success_rate"] > 0)
    n_all = sum(1 for r in results if r["success_rate"] == 1.0)
    n_total_episodes = sum(len(v) for v in rollouts_by_instance.values())

    summary = {
        "dataset": args.dataset, "model": args.model, "n_runs": args.n_runs,
        "n_instances": n_instances, "n_total_episodes": n_total_episodes,
        "overall_success_rate": avg_success,
        "per_instance": {
            r["instance_id"]: {
                "success_rate": r["success_rate"],
                "n_runs_completed": r["n_runs_completed"],
            }
            for r in results
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAvg success rate: {avg_success:.1%}")
    print(f"pass@{args.n_runs}: {n_any}/{n_instances} ({100*n_any/n_instances:.1f}%)")
    print(f"cons@{args.n_runs}: {n_all}/{n_instances} ({100*n_all/n_instances:.1f}%)")
    print(f"Total episodes: {n_total_episodes}")
    print(f"Saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
