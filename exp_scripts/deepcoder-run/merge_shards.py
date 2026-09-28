"""Reassemble sharded DeepCoder eval shards into one 687-task result.

Every check is fatal. A merge that silently accepts a missing or duplicated task
would report a success rate over the wrong denominator, which is exactly the kind
of error that looks like a real effect.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

EXPECTED_TASKS = 687


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", action="append", required=True, help="a shard run dir (repeatable)")
    ap.add_argument("--output", required=True, help="dir to write the merged result.json")
    args = ap.parse_args()

    from rllm.eval.results import EvalItem, EvalResult

    items: list[EvalItem] = []
    protocols = []
    for d in args.shard:
        p = Path(d)
        rp, pp = p / "result.json", p / "protocol.json"
        for f in (rp, pp):
            if not f.exists():
                raise RuntimeError(f"missing {f}; shard {p} did not complete")
        proto = json.loads(pp.read_text())
        protocols.append(proto)
        data = json.loads(rp.read_text())
        if len(data["items"]) != proto["tasks"]:
            raise RuntimeError(f"{p}: result has {len(data['items'])} items, protocol declares {proto['tasks']}")
        items.extend(
            EvalItem(idx=i["idx"], reward=i["reward"], is_correct=i["is_correct"],
                     error=i.get("error"), signals=i.get("signals", {}), attempt=i.get("attempt", 0))
            for i in data["items"]
        )

    # The shards must agree on everything that defines the measurement.
    keys = ("model", "sampling", "split", "concurrency", "max_model_len", "num_shards")
    first = {k: protocols[0].get(k) for k in keys}
    for proto, d in zip(protocols[1:], args.shard[1:], strict=True):
        other = {k: proto.get(k) for k in keys}
        if other != first:
            raise RuntimeError(f"protocol mismatch between {args.shard[0]} and {d}:\n  {first}\n  {other}")
    seen_shards = sorted(p["shard_index"] for p in protocols)
    if seen_shards != list(range(first["num_shards"])):
        raise RuntimeError(f"expected shard indices {list(range(first['num_shards']))}, got {seen_shards}")

    idxs = sorted(i.idx for i in items)
    if idxs != list(range(EXPECTED_TASKS)):
        missing = sorted(set(range(EXPECTED_TASKS)) - set(idxs))
        dupes = sorted({i for i in idxs if idxs.count(i) > 1})
        raise RuntimeError(
            f"merged item set is not exactly 0..{EXPECTED_TASKS - 1}: "
            f"{len(idxs)} items, {len(missing)} missing {missing[:10]}, duplicated {dupes[:10]}"
        )

    items.sort(key=lambda i: i.idx)
    merged = EvalResult.from_items("deepcoder", protocols[0]["model"], "deepcoder", items, attempts=1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(asdict(merged), indent=2))
    (out / "protocol.json").write_text(json.dumps(
        {**{k: protocols[0][k] for k in keys if k in protocols[0]},
         "tasks": EXPECTED_TASKS, "merged_from": [str(Path(d).resolve()) for d in args.shard]}, indent=2))
    print(f"[merge] {EXPECTED_TASKS} tasks from {len(args.shard)} shards -> "
          f"{100 * merged.correct / merged.total:.2f}% ({merged.correct}/{merged.total}) errors={merged.errors}")


if __name__ == "__main__":
    main()
