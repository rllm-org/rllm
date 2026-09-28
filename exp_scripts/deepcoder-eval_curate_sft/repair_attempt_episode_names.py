"""Repair pre-fix eval files whose names use attempt-qualified Episode ids.

The operation is fail-closed and content preserving: it validates every
attempt group before renaming files, and refuses collisions or ambiguity.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


EPISODE_RE = re.compile(r"^episode_(\d+)_(.+)_(\d+)\.json$")


def main(run_dir: Path, attempts: int, apply: bool) -> None:
    episodes_dir = run_dir / "episodes"
    files = sorted(episodes_dir.glob("episode_*.json"))
    if not files:
        raise RuntimeError(f"no episodes found under {episodes_dir}")

    groups: dict[int, list[tuple[Path, int, str, int]]] = defaultdict(list)
    for path in files:
        match = EPISODE_RE.match(path.name)
        if not match:
            raise RuntimeError(f"unexpected episode filename: {path.name}")
        eval_idx = int(match.group(1))
        stable_id = match.group(2)
        suffix_attempt = int(match.group(3))
        task_pos, expected_attempt = divmod(eval_idx, attempts)
        if suffix_attempt != expected_attempt:
            raise RuntimeError(
                f"{path.name}: suffix attempt {suffix_attempt} != "
                f"eval index attempt {expected_attempt}"
            )
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        embedded = payload.get("task") or {}
        embedded_id = embedded.get("task_id") if isinstance(embedded, dict) else None
        if embedded_id != stable_id:
            raise RuntimeError(
                f"{path.name}: embedded task_id {embedded_id!r} != {stable_id!r}"
            )
        groups[task_pos].append((path, eval_idx, stable_id, suffix_attempt))

    expected_slots = list(range(attempts))
    renames: list[tuple[Path, Path]] = []
    for task_pos, group in sorted(groups.items()):
        ids = {entry[2] for entry in group}
        slots = sorted(entry[3] for entry in group)
        if len(ids) != 1 or slots != expected_slots:
            raise RuntimeError(
                f"task position {task_pos}: ids={sorted(ids)}, slots={slots}, "
                f"expected slots={expected_slots}"
            )
        stable_id = next(iter(ids))
        for source, eval_idx, _, _ in group:
            target = episodes_dir / f"episode_{eval_idx:06d}_{stable_id}.json"
            if target.exists() and target != source:
                raise RuntimeError(f"refusing overwrite: {target}")
            renames.append((source, target))

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "tasks": len(groups),
                "episodes": len(renames),
                "attempts_per_task": attempts,
                "mode": "apply" if apply else "dry-run",
            },
            indent=2,
        )
    )
    if apply:
        for source, target in renames:
            source.rename(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    ns = parser.parse_args()
    if ns.attempts < 1:
        parser.error("--attempts must be positive")
    main(ns.run_dir.expanduser().resolve(), ns.attempts, ns.apply)
