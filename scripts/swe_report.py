#!/usr/bin/env python3
"""
SWE Evaluation Report Script

Reads per-instance validation results (JSONL) produced by the eval pipeline
and generates comprehensive statistics.

Usage:
    python scripts/swe_report.py <results_jsonl>
    python scripts/swe_report.py /path/to/val_results/step_0.jsonl

The JSONL file is saved by _validate_agent() in:
    {default_local_dir}/val_results/step_{global_steps}.jsonl
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_results(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    Args:
        n: total number of samples per task
        c: number of correct samples for the task
        k: k value for pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def report_single_sample(records: list[dict]):
    """Report for n=1: dataset-level statistics."""
    rewards = [r["reward"] for r in records]
    n_total = len(rewards)
    n_solved = sum(1 for r in rewards if r >= 1.0)
    n_failed = n_total - n_solved
    mean = sum(rewards) / n_total if n_total > 0 else 0.0
    std = (sum((r - mean) ** 2 for r in rewards) / n_total) ** 0.5 if n_total > 0 else 0.0

    print("=" * 60)
    print("  SWE Evaluation Report (n=1, greedy)")
    print("=" * 60)
    print()
    print(f"  Total instances:   {n_total}")
    print(f"  Resolved:          {n_solved}  ({n_solved / n_total * 100:.1f}%)")
    print(f"  Failed:            {n_failed}  ({n_failed / n_total * 100:.1f}%)")
    print()
    print(f"  Resolve rate:      {mean:.4f}")
    print(f"  Std dev:           {std:.4f}")
    print()

    # Per-repo breakdown
    repo_results = defaultdict(list)
    for r in records:
        repo = r.get("repo", "unknown")
        repo_results[repo].append(r["reward"])

    if len(repo_results) > 1:
        print("-" * 60)
        print("  Per-repo breakdown")
        print("-" * 60)
        print(f"  {'Repo':<40s} {'Solved':>7s} {'Total':>6s} {'Rate':>8s}")
        print(f"  {'----':<40s} {'------':>7s} {'-----':>6s} {'----':>8s}")

        sorted_repos = sorted(repo_results.items(), key=lambda x: -sum(1 for v in x[1] if v >= 1.0) / len(x[1]))
        for repo, rews in sorted_repos:
            repo_n = len(rews)
            repo_solved = sum(1 for v in rews if v >= 1.0)
            repo_rate = repo_solved / repo_n if repo_n > 0 else 0.0
            display_repo = repo if len(repo) <= 38 else "..." + repo[-35:]
            print(f"  {display_repo:<40s} {repo_solved:>7d} {repo_n:>6d} {repo_rate:>7.1%}")

        print()

    # Solved instance list
    print("-" * 60)
    print("  Resolved instances")
    print("-" * 60)
    solved_ids = sorted([r.get("instance_id", r.get("uid", "?")) for r in records if r["reward"] >= 1.0])
    for sid in solved_ids:
        print(f"    {sid}")
    if not solved_ids:
        print("    (none)")
    print()


def report_multi_sample(records: list[dict]):
    """Report for n>1: dataset-level + sample-level statistics."""
    n_samples = records[0].get("n_samples", 1)

    # Group by uid (same instance across samples)
    uid_groups = defaultdict(list)
    for r in records:
        uid_groups[r["uid"]].append(r)

    n_instances = len(uid_groups)
    # Per-instance: count correct samples
    instance_stats = []
    for uid, samples in uid_groups.items():
        samples_sorted = sorted(samples, key=lambda x: x.get("sample_idx", 0))
        c = sum(1 for s in samples_sorted if s["reward"] >= 1.0)
        n = len(samples_sorted)
        instance_stats.append({
            "uid": uid,
            "instance_id": samples_sorted[0].get("instance_id", ""),
            "repo": samples_sorted[0].get("repo", "unknown"),
            "n": n,
            "c": c,
            "mean_reward": sum(s["reward"] for s in samples_sorted) / n,
            "rewards": [s["reward"] for s in samples_sorted],
        })

    # Dataset-level
    all_rewards = [r["reward"] for r in records]
    overall_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    overall_std = (sum((r - overall_mean) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5 if all_rewards else 0.0

    # pass@k for various k values
    k_values = sorted(set([1, min(n_samples, 5), n_samples]))
    k_values = [k for k in k_values if k <= n_samples]

    pass_at_k_results = {}
    for k in k_values:
        scores = [pass_at_k(inst["n"], inst["c"], k) for inst in instance_stats]
        pass_at_k_results[k] = sum(scores) / len(scores) if scores else 0.0

    # Per-instance mean rewards (for error bar analysis)
    instance_means = [inst["mean_reward"] for inst in instance_stats]
    instance_means_mean = sum(instance_means) / len(instance_means) if instance_means else 0.0
    instance_means_std = (sum((m - instance_means_mean) ** 2 for m in instance_means) / len(instance_means)) ** 0.5 if instance_means else 0.0
    # Standard error of the mean (for confidence intervals)
    sem = instance_means_std / (len(instance_means) ** 0.5) if instance_means else 0.0

    print("=" * 60)
    print(f"  SWE Evaluation Report (n={n_samples}, sampling)")
    print("=" * 60)
    print()

    # --- Dataset-level ---
    print(f"  Total instances:     {n_instances}")
    print(f"  Samples per instance:{n_samples}")
    print(f"  Total samples:       {len(records)}")
    print()
    print("  --- Dataset-level (across all samples) ---")
    print(f"  Mean reward:         {overall_mean:.4f}")
    print(f"  Std dev:             {overall_std:.4f}")
    print()

    # --- pass@k ---
    print("  --- pass@k (unbiased estimator) ---")
    for k, score in pass_at_k_results.items():
        print(f"  pass@{k:<3d}             {score:.4f}  ({score * 100:.1f}%)")
    print()

    # --- Instance-level (for error bars) ---
    print("  --- Instance-level statistics (for error bars) ---")
    print(f"  Mean of per-instance means:  {instance_means_mean:.4f}")
    print(f"  Std of per-instance means:   {instance_means_std:.4f}")
    print(f"  SEM (standard error):        {sem:.4f}")
    print(f"  95% CI:                      {instance_means_mean:.4f} +/- {1.96 * sem:.4f}")
    print()

    # Distribution of per-instance solve counts
    solve_count_dist = defaultdict(int)
    for inst in instance_stats:
        solve_count_dist[inst["c"]] += 1

    print("  --- Solve count distribution ---")
    print(f"  {'Solved k/' + str(n_samples):>15s}  {'Instances':>10s}  {'Fraction':>10s}")
    print(f"  {'---------------':>15s}  {'----------':>10s}  {'----------':>10s}")
    for k in range(n_samples + 1):
        count = solve_count_dist.get(k, 0)
        frac = count / n_instances if n_instances > 0 else 0.0
        bar = "#" * int(frac * 30)
        print(f"  {k:>15d}  {count:>10d}  {frac:>9.1%}  {bar}")
    print()

    # Per-repo breakdown
    repo_instances = defaultdict(list)
    for inst in instance_stats:
        repo_instances[inst["repo"]].append(inst)

    if len(repo_instances) > 1:
        print("-" * 60)
        print("  Per-repo breakdown")
        print("-" * 60)
        header_k = "  ".join([f"p@{k}" for k in k_values])
        print(f"  {'Repo':<35s} {'N':>4s}  {header_k}")
        print(f"  {'----':<35s} {'--':>4s}  {'  '.join(['----'] * len(k_values))}")

        sorted_repos = sorted(repo_instances.items(), key=lambda x: -pass_at_k_results.get(1, 0))
        for repo, insts in sorted_repos:
            repo_n = len(insts)
            repo_pass_k = []
            for k in k_values:
                scores = [pass_at_k(inst["n"], inst["c"], k) for inst in insts]
                repo_pass_k.append(sum(scores) / len(scores) if scores else 0.0)
            k_str = "  ".join([f"{v:.1%}" for v in repo_pass_k])
            display_repo = repo if len(repo) <= 33 else "..." + repo[-30:]
            print(f"  {display_repo:<35s} {repo_n:>4d}  {k_str}")

        print()

    # Resolved instances (at least one sample solved)
    print("-" * 60)
    print("  Instances resolved (at least 1 sample correct)")
    print("-" * 60)
    solved_insts = sorted(
        [inst for inst in instance_stats if inst["c"] > 0],
        key=lambda x: -x["c"],
    )
    for inst in solved_insts:
        iid = inst["instance_id"] or inst["uid"]
        print(f"    {iid:<50s}  {inst['c']}/{inst['n']} correct")
    if not solved_insts:
        print("    (none)")
    print()


def main():
    parser = argparse.ArgumentParser(description="SWE Evaluation Report")
    parser.add_argument("results", help="Path to val_results JSONL file")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    records = load_results(str(path))
    if not records:
        print("Error: no records found", file=sys.stderr)
        sys.exit(1)

    n_samples = records[0].get("n_samples", 1)

    if n_samples <= 1:
        report_single_sample(records)
    else:
        report_multi_sample(records)


if __name__ == "__main__":
    main()
