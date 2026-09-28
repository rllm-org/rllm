"""Fail-closed validation for deterministic multi-GPU eval shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_meta(path: str) -> dict:
    run_dir = Path(path).expanduser().resolve()
    with (run_dir / "meta.json").open() as handle:
        meta = json.load(handle)
    if not (run_dir / "results.json").is_file():
        raise RuntimeError(f"missing results.json in {run_dir}")
    return meta


def main(args: argparse.Namespace) -> None:
    metas = [load_meta(path) for path in args.run_dir]
    source_sizes = {int(meta["source_size"]) for meta in metas}
    if len(source_sizes) != 1:
        raise RuntimeError(f"shards disagree on source_size: {sorted(source_sizes)}")
    source_size = source_sizes.pop()

    index_sets = [set(int(i) for i in meta["selected_source_indices"]) for meta in metas]
    for i, left in enumerate(index_sets):
        for j, right in enumerate(index_sets[i + 1 :], start=i + 1):
            overlap = left & right
            if overlap:
                raise RuntimeError(f"shards {i} and {j} overlap on {len(overlap)} tasks")

    union = set().union(*index_sets)
    if args.require_full and union != set(range(source_size)):
        missing = source_size - len(union)
        raise RuntimeError(
            f"full-shard coverage failed: source={source_size}, union={len(union)}, "
            f"missing_or_extra_delta={missing}"
        )

    print(
        json.dumps(
            {
                "source_size": source_size,
                "shard_sizes": [len(indices) for indices in index_sets],
                "union_size": len(union),
                "overlap": 0,
                "full_coverage": union == set(range(source_size)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--require-full", action="store_true")
    main(parser.parse_args())
