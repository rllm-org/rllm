"""Create the fixed 48-task OpenCode curriculum from completed rollout logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rllm.data import DatasetRegistry

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPISODE_RE = re.compile(
    r"\[([^\]\n]+):(\d+)\] Rewards: \[opencode: ([01](?:\.0)?)\].*"
    r"Termination: TerminationReason\.([A-Z_]+)"
)


@dataclass(frozen=True)
class Candidate:
    task_id: str
    successes: tuple[int, ...]


def clean_medium_candidates(log_paths: Path | list[Path], allowed_task_ids: set[str] | None = None) -> list[Candidate]:
    if isinstance(log_paths, Path):
        log_paths = [log_paths]
    observations: dict[str, list[tuple[int, bool]]] = {}
    expected_indices = set(range(8))
    for log_path in log_paths:
        text = ANSI_RE.sub("", log_path.read_text(errors="ignore")).replace("\r", "\n")
        grouped: dict[str, dict[int, tuple[float, str]]] = {}
        for match in EPISODE_RE.finditer(text):
            task_id, rollout_index, reward, termination = match.groups()
            if allowed_task_ids is not None and task_id not in allowed_task_ids:
                continue
            index = int(rollout_index)
            result = (float(reward), termination)
            previous = grouped.setdefault(task_id, {}).get(index)
            if previous is not None and previous != result:
                raise RuntimeError(f"Conflicting results for {task_id} rollout {index} within {log_path}")
            grouped[task_id][index] = result

        for task_id, rollouts in grouped.items():
            if set(rollouts) != expected_indices:
                continue
            results = [rollouts[index] for index in range(8)]
            successes = int(sum(reward for reward, _ in results))
            clean = all(termination == "ENV_DONE" for _, termination in results)
            observations.setdefault(task_id, []).append((successes, clean))

    return [
        Candidate(task_id=task_id, successes=tuple(successes for successes, _ in task_observations))
        for task_id, task_observations in observations.items()
        if all(3 <= successes <= 5 and clean for successes, clean in task_observations)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path, action="append")
    parser.add_argument("--source-dataset", default="tb-opus-pass")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--output-dataset", default="tb-opencode-medium-48")
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", default="20260727")
    args = parser.parse_args()

    source = DatasetRegistry.load_dataset(args.source_dataset, args.source_split)
    if source is None:
        raise RuntimeError(f"Source dataset {args.source_dataset}/{args.source_split} is not registered")
    rows_by_id = {str(row["task_id"]): row for row in source.get_data()}

    candidates = clean_medium_candidates(args.log, allowed_task_ids=set(rows_by_id))
    if len(candidates) < args.count:
        raise RuntimeError(f"Only {len(candidates)} clean medium tasks found; need {args.count}")

    ranked = sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(f"{args.seed}:{candidate.task_id}".encode()).digest(),
    )
    selected = ranked[: args.count]

    missing = sorted(candidate.task_id for candidate in selected if candidate.task_id not in rows_by_id)
    if missing:
        raise RuntimeError(f"{len(missing)} selected task IDs are absent from the source dataset: {missing[:3]}")

    dataset = DatasetRegistry.register_dataset(
        name=args.output_dataset,
        data=[rows_by_id[candidate.task_id] for candidate in selected],
        split="train",
        source=f"{args.source_dataset}/{args.source_split} curated from {','.join(log.name for log in args.log)}",
        description="Fixed 48-task OpenCode curriculum: 3-5/8 successes and eight ENV_DONE rollouts",
        category="agentic",
    )
    data_path = Path(dataset.get_data_path() or "")
    manifest_path = data_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source_logs": [str(log.resolve()) for log in args.log],
                "seed": args.seed,
                "eligible_count": len(candidates),
                "selected": [asdict(candidate) for candidate in selected],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"registered {args.output_dataset}/train with {len(selected)} tasks")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
