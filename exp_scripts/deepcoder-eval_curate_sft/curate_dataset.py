"""Curate successful DeepCoder episodes into leak-audited SFT parquet files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from rllm.data import DatasetRegistry
from rllm.eval.curation import CurationConfig, curate
from rllm.eval.results import EvalResult


PRIVATE_FIELDS = ("tests", "ground_truth", "solutions")


def string_leaves(value) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in string_leaves(child)]
    if isinstance(value, (list, tuple)):
        return [leaf for child in value for leaf in string_leaves(child)]
    if value is None:
        return []
    return [str(value)]


def private_strings(row: dict) -> list[str]:
    values: list[str] = []
    for key in PRIVATE_FIELDS:
        values.extend(string_leaves(row.get(key)))
    # Avoid meaningless matches on small scalars/fragments.
    return [x for x in values if len(x.strip()) >= 32]


def normalized_text(value) -> str:
    """Flatten visible message values while preserving multiline payloads."""
    return " ".join(" ".join(string_leaves(value)).split())


def rendered_length(tokenizer, messages: list[dict]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # Transformers versions differ: this may be a bare list of token ids or a
    # BatchEncoding. ``len(BatchEncoding)`` is the number of fields (typically
    # 2), not the token count, so always unwrap input_ids when present.
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "shape"):
        shape = tuple(encoded.shape)
        return int(shape[-1]) if shape else 0
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise RuntimeError(f"expected one rendered conversation, got {len(encoded)}")
        encoded = encoded[0]
    return len(encoded)


def build_coverage_probe(run_dirs: list[Path], expected_attempts: int = 8) -> dict:
    """Measure task coverage before curation from verifier outcomes."""
    histogram = {successes: 0 for successes in range(expected_attempts + 1)}
    tasks_with_rollout_errors = 0
    rollout_errors = 0
    tasks = 0
    rollouts = 0
    for run_dir in run_dirs:
        result = EvalResult.load(str(run_dir / "results.json"))
        if result.attempts != expected_attempts:
            raise RuntimeError(
                f"coverage probe expected {expected_attempts} attempts/task, "
                f"{run_dir}/results.json declares {result.attempts}"
            )

        grouped: dict[int, list] = defaultdict(list)
        for item in result.items:
            grouped[item.idx].append(item)
        incomplete = {idx: len(items) for idx, items in grouped.items() if len(items) != expected_attempts}
        if incomplete:
            preview = dict(list(sorted(incomplete.items()))[:10])
            raise RuntimeError(f"incomplete attempt groups in {run_dir}/results.json: {preview}")

        tasks += len(grouped)
        rollouts += sum(len(items) for items in grouped.values())
        for items in grouped.values():
            successes = sum(bool(item.is_correct) for item in items)
            histogram[successes] += 1
            errors = sum(item.error is not None for item in items)
            rollout_errors += errors
            tasks_with_rollout_errors += int(errors > 0)

    def share(count: int, denominator: int) -> float:
        return count / denominator if denominator else 0.0

    zero_success = histogram[0]
    all_success = histogram[expected_attempts]
    partial_success = tasks - zero_success - all_success
    return {
        "source_runs": [str(path) for path in run_dirs],
        "attempts_per_task": expected_attempts,
        "tasks": tasks,
        "rollouts": rollouts,
        "success_count_histogram": {str(k): v for k, v in histogram.items()},
        "zero_success_tasks": zero_success,
        "zero_success_task_share": share(zero_success, tasks),
        "partial_success_tasks_eligible_before_audit": partial_success,
        "partial_success_task_share": share(partial_success, tasks),
        "all_success_tasks_eligible_before_audit": all_success,
        "all_success_task_share": share(all_success, tasks),
        "rollout_errors": rollout_errors,
        "rollout_error_share": share(rollout_errors, rollouts),
        "tasks_with_rollout_errors": tasks_with_rollout_errors,
        "tasks_with_rollout_errors_share": share(tasks_with_rollout_errors, tasks),
    }


def main(args: argparse.Namespace) -> None:
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dir]
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "train.jsonl",
        "val.jsonl",
        "train.parquet",
        "val.parquet",
        "curation_manifest.json",
        "coverage_probe.json",
    ):
        if (out / name).exists():
            raise RuntimeError(f"refusing to overwrite {out / name}")

    coverage = build_coverage_probe(run_dirs, expected_attempts=8)
    # Persist coverage even if the later difficulty filter or leakage audit
    # intentionally fails closed (especially useful for a small smoke sample).
    (out / "coverage_probe.json").write_text(json.dumps(coverage, indent=2) + "\n")

    rows, stats = curate(
        run_dirs,
        CurationConfig(
            metric="is_correct",
            filter_expr="avg > 0",
            select="shortest",
            max_per_task=args.max_select_rollouts_cnt,
            # Preserve the requested per-task contribution count. Identical
            # successful attempts are still distinct sampled trajectories.
            dedup=False,
        ),
    )

    source = DatasetRegistry.load_dataset("deepcoder", "train")
    if source is None:
        raise RuntimeError("DeepCoder train split is not registered")
    source_by_uid = {
        str(row.get("uid") or row.get("index") or idx): dict(row)
        for idx, row in enumerate(source.data)
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    accepted: list[dict] = []
    overlength = 0
    lengths: list[int] = []
    for row in rows:
        task_id = str(row["task_id"])
        messages = row["messages"]
        if task_id not in source_by_uid:
            raise RuntimeError(f"curated task {task_id!r} cannot be mapped to source data")
        if not messages or messages[-1].get("role") != "assistant":
            raise RuntimeError(f"task {task_id}: conversation lacks final assistant message")
        if not str(messages[-1].get("content") or "").strip():
            raise RuntimeError(f"task {task_id}: empty successful assistant message")

        # Private verifier/reference material must never appear in what the
        # policy was conditioned on.  Do not compare it against assistant
        # labels: for simple deterministic problems a sampled, verifier-passing
        # answer can legitimately be byte-identical to a short reference
        # solution.  That is successful self-generated supervision, not input
        # leakage.
        policy_input_text = normalized_text(
            [message for message in messages if message.get("role") != "assistant"]
        )
        for private in private_strings(source_by_uid[task_id]):
            private_text = normalized_text(private)
            if private_text and private_text in policy_input_text:
                raise RuntimeError(
                    f"task {task_id}: private verifier/reference content leaked into policy input"
                )

        length = rendered_length(tokenizer, messages)
        if length > args.max_length:
            overlength += 1
            continue
        lengths.append(length)
        # Only messages cross the SFT boundary.  Reward and correctness remain
        # in the audit manifest/statistics, never in policy-visible parquet.
        accepted.append({"task_id": task_id, "messages": messages, "length": length})

    if not accepted:
        raise RuntimeError("curation produced no leak-free, in-budget SFT rows")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in accepted:
        by_task[row["task_id"]].append(row)
    contribution_histogram = Counter(len(task_rows) for task_rows in by_task.values())
    task_ids = sorted(by_task)
    rng = random.Random(args.split_seed)
    rng.shuffle(task_ids)
    n_val = int(round(len(task_ids) * args.val_fraction))
    if args.val_fraction > 0 and len(task_ids) > 1:
        n_val = max(1, min(n_val, len(task_ids) - 1))
    val_ids = set(task_ids[:n_val])

    train_rows = [{"messages": r["messages"]} for r in accepted if r["task_id"] not in val_ids]
    val_rows = [{"messages": r["messages"]} for r in accepted if r["task_id"] in val_ids]
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(train_rows).to_parquet(out / "train.parquet", index=False)
    if val_rows:
        with open(out / "val.jsonl", "w", encoding="utf-8") as f:
            for row in val_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pd.DataFrame(val_rows).to_parquet(out / "val.parquet", index=False)

    q = np.percentile(lengths, [50, 90, 95, 99, 100]).tolist()
    manifest = {
        "source_runs": [str(path) for path in run_dirs],
        "filter": "avg(is_correct) > 0",
        "selection": (
            "shortest correct trajectories, up to "
            f"min({args.max_select_rollouts_cnt}, successful_rollouts) per task"
        ),
        "max_select_rollouts_cnt": args.max_select_rollouts_cnt,
        "dedup": False,
        "max_length": args.max_length,
        "model": args.model,
        "tasks_total": stats.tasks_total,
        "tasks_kept_with_success": stats.tasks_kept,
        "candidate_rows": stats.rows_emitted,
        "rows_dropped_overlength": overlength,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "accepted_rows_per_task_histogram": {
            str(count): tasks for count, tasks in sorted(contribution_histogram.items())
        },
        "split_seed": args.split_seed,
        "token_lengths": dict(zip(("p50", "p90", "p95", "p99", "max"), q, strict=True)),
        "sft_input_format": "JSONL (Parquet retained as an audit copy)",
        "sft_policy_visible_columns": ["messages"],
        "private_fields_checked_against": "policy input messages (all non-assistant roles)",
        "private_fields_checked": list(PRIVATE_FIELDS),
        "coverage_probe": coverage,
    }
    (out / "curation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-select-rollouts-cnt", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=1234)
    args = parser.parse_args()
    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1)")
    if not 1 <= args.max_select_rollouts_cnt <= 8:
        parser.error("--max-select-rollouts-cnt must be in [1, 8]")
    main(args)
