"""Compare rLLM MCP-Atlas results with an official scored CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def load_rllm(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["task_id"]): float(item.get("signals", {}).get("coverage", 0.0)) for item in data["items"] if item.get("task_id") is not None and item.get("error") is None}


def load_official(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    output = {}
    for row in rows:
        task_id = row.get("TASK") or row.get("task_id")
        score = row.get("coverage_score")
        if task_id and score not in (None, ""):
            output[str(task_id)] = float(score)
    return output


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * proportion)]


def paired_bootstrap_ci(differences: list[float], *, samples: int = 10_000, seed: int = 20260806) -> tuple[float, float]:
    rng = random.Random(seed)
    means = [sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_mean_ci(values: list[float], *, samples: int = 10_000, seed: int = 20260806) -> tuple[float, float]:
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def compare(rllm: dict[str, float], official: dict[str, float], threshold: float = 0.75) -> dict:
    task_ids = sorted(set(rllm) & set(official))
    if not task_ids:
        raise ValueError("No common task IDs between rLLM and official results")
    score_differences = [rllm[task_id] - official[task_id] for task_id in task_ids]
    rllm_coverage = [rllm[task_id] for task_id in task_ids]
    official_coverage = [official[task_id] for task_id in task_ids]
    rllm_passes = [float(score >= threshold) for score in rllm_coverage]
    official_passes = [float(score >= threshold) for score in official_coverage]
    pass_differences = [float(rllm[task_id] >= threshold) - float(official[task_id] >= threshold) for task_id in task_ids]
    ci_low, ci_high = paired_bootstrap_ci(pass_differences)
    rllm_pass_rate = sum(rllm_passes) / len(task_ids)
    official_pass_rate = sum(official_passes) / len(task_ids)
    return {
        "task_count": len(task_ids),
        "missing_from_rllm": sorted(set(official) - set(rllm)),
        "missing_from_official": sorted(set(rllm) - set(official)),
        "exact_coverage_matches": sum(abs(value) < 1e-12 for value in score_differences),
        "max_absolute_coverage_difference": max(abs(value) for value in score_differences),
        "rllm_mean_coverage": sum(rllm_coverage) / len(task_ids),
        "rllm_mean_coverage_95_ci": list(bootstrap_mean_ci(rllm_coverage, seed=20260806)),
        "official_mean_coverage": sum(official_coverage) / len(task_ids),
        "official_mean_coverage_95_ci": list(bootstrap_mean_ci(official_coverage, seed=20260807)),
        "rllm_pass_rate": rllm_pass_rate,
        "rllm_pass_rate_95_ci": list(bootstrap_mean_ci(rllm_passes, seed=20260808)),
        "official_pass_rate": official_pass_rate,
        "official_pass_rate_95_ci": list(bootstrap_mean_ci(official_passes, seed=20260809)),
        "pass_rate_difference": rllm_pass_rate - official_pass_rate,
        "paired_bootstrap_95_ci": [ci_low, ci_high],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rllm", type=Path, required=True, help="rLLM results.json")
    parser.add_argument("--official", type=Path, required=True, help="Official score_claims.py output CSV")
    parser.add_argument("--mode", choices=("replay", "live"), default="live")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare(load_rllm(args.rllm), load_official(args.official))
    if args.mode == "replay":
        report["accepted"] = not report["missing_from_rllm"] and not report["missing_from_official"] and report["max_absolute_coverage_difference"] == 0.0
    else:
        low, high = report["paired_bootstrap_95_ci"]
        report["accepted"] = abs(report["pass_rate_difference"]) <= 0.05 and low <= 0.0 <= high
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
