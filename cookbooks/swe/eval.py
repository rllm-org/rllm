#!/usr/bin/env python3
"""Run SWE-bench evaluation using the AgentFlow/Evaluator framework.

Usage:
    # OpenAI
    python eval.py \
        --base-url https://api.openai.com/v1 \
        --model gpt-4.1-mini \
        --dataset swe_bench_pro --slice 0:5 \
        --output_dir results/test/ -v

    # Self-hosted vLLM
    python eval.py \
        --base-url http://localhost:8000/v1 \
        --api-key EMPTY \
        --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --dataset swe_smith_py --n_parallel 50 \
        --output_dir results/qwen/

    # Tinker (start proxy first: # Start tinker proxy first --model Qwen/Qwen3-8B)
    python eval.py \
        --base-url http://127.0.0.1:4123/v1 \
        --api-key EMPTY \
        --model Qwen/Qwen3-8B \
        --dataset swe_bench_pro --slice 0:5 \
        --output_dir results/tinker/
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

_COOKBOOK_DIR = Path(__file__).resolve().parent

from dotenv import load_dotenv

from data import prepare_dataset, load_prepared_data, DATASET_CONFIGS

load_dotenv(_COOKBOOK_DIR / ".env", verbose=False)


def _clean_messages(messages: list[dict]) -> list[dict]:
    """Strip bulky metadata from messages for trajectory saving."""
    cleaned = []
    for msg in messages:
        m = {"role": msg.get("role", "unknown"), "content": msg.get("content", "")}
        extra = msg.get("extra", {})
        if extra:
            # Keep useful fields, drop raw API response
            clean_extra = {}
            for key in ("actions", "format_error", "exit_status", "submission",
                        "interrupt_type", "model_response", "returncode", "raw_output"):
                if key in extra:
                    clean_extra[key] = extra[key]
            if clean_extra:
                m["extra"] = clean_extra
        cleaned.append(m)
    return cleaned


async def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation",
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
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible endpoint (e.g. https://api.openai.com/v1)")
    parser.add_argument("--api-key", default=None, help="API key (default: OPENAI_API_KEY env var)")
    parser.add_argument("--model", required=True, help="Model name")

    # Limits
    parser.add_argument("--cost_limit", type=float, default=50.0)
    parser.add_argument("--n_parallel", type=int, default=50)
    parser.add_argument("--slice", default="")
    parser.add_argument("--instance_ids", nargs="+", default=None)
    parser.add_argument("--step_limit", type=int, default=50)
    parser.add_argument("--command_timeout", type=int, default=60)
    parser.add_argument("--sandbox_timeout", type=int, default=3600)
    parser.add_argument("--agent_timeout", type=int, default=750)

    # Output
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")

    # SWE-bench Pro
    parser.add_argument("--scripts_dir", default=None)
    parser.add_argument("--dockerhub_username", default="jefzda")

    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # --- Output dir ---
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _COOKBOOK_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if args.data_path:
        print(f"Loading pre-prepared data from {args.data_path}")
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

    print(f"\nRunning on {len(tasks)} tasks from {args.dataset}")
    print(f"  base_url: {args.base_url}")
    print(f"  model:    {args.model}")
    print(f"  parallel: {args.n_parallel}")
    print(f"  output:   {output_dir}")

    # --- Create AgentFlow + Evaluator ---
    from swe_agent_flow import SWEAgentFlow
    from evaluator import SWEEvaluator
    from rllm.experimental.eval.runner import EvalRunner

    agent = SWEAgentFlow(
        cost_limit=args.cost_limit,
        step_limit=args.step_limit,
        command_timeout=args.command_timeout,
        sandbox_timeout=args.sandbox_timeout,
        agent_timeout=args.agent_timeout,
        verbose=args.verbose,
    )
    evaluator = SWEEvaluator(
        scripts_dir=args.scripts_dir,
        dockerhub_username=args.dockerhub_username,
        command_timeout=args.command_timeout,
        sandbox_timeout=args.sandbox_timeout,
        verbose=args.verbose,
    )
    runner = EvalRunner(
        base_url=args.base_url,
        model=args.model,
        concurrency=args.n_parallel,
    )

    # --- Run ---
    eval_result, episodes = await runner.run(tasks, agent, evaluator, agent_name="swe")

    # --- Save results + trajectories ---
    results = []
    patches = {}

    for episode in episodes:
        if episode is None:
            continue
        instance_id = episode.task["instance_id"]
        patch = episode.artifacts["patch"]
        exit_status = episode.artifacts["exit_status"]
        messages = episode.artifacts["messages"]

        result = {
            "instance_id": instance_id,
            "is_correct": episode.is_correct,
            "patch_length": len(patch),
            "exit_status": exit_status,
        }
        if patch:
            result["patch"] = patch
            patches[instance_id] = {
                "instance_id": instance_id,
                "model_name_or_path": args.model,
                "model_patch": patch,
            }
        results.append(result)

        # Save trajectory
        if messages:
            safe_id = instance_id.replace("/", "_").replace(":", "_")
            traj_file = trajectories_dir / f"{safe_id}.json"
            with open(traj_file, "w") as f:
                json.dump(_clean_messages(messages), f, indent=2)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "preds.json", "w") as f:
        json.dump(patches, f, indent=2)
    with open(output_dir / "preds_list.json", "w") as f:
        json.dump(list(patches.values()), f, indent=2)

    # --- Summary ---
    n_total = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    n_patches = sum(1 for r in results if r.get("patch"))

    print(f"\n{'='*50}")
    print(f"Results: {n_correct}/{n_total} correct, {n_patches} patches")
    print(f"EvalResult: {eval_result.score:.1%} accuracy")
    print(f"Saved to {output_dir}/")
    print(f"  trajectories/ ({len([e for e in episodes if e])} files)")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
