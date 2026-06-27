"""Smoke-check tmax-15k tasks on Daytona.

For each task: boot its image, run test_initial_state (pristine env, should pass)
and test_final_state (no agent acted, reward should be 0.0), write one JSONL line.

test_final_state + image/resources come from the materialized training dataset and
run through training's own ShellScriptEvaluator. test_initial_state lives only in
the raw HF corpus (cached locally by `rllm dataset pull`), read offline.

    python cookbooks/tmax/sanity.py --limit 5 --concurrency 5 -o /tmp/smoke.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Read the cached HF corpus without contacting the Hub (set before datasets import).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

DATASET_DIR = Path.home() / ".rllm" / "datasets" / "tmax-15k"
HF_REPO = "allenai/TMax-15K"

_INIT = """set -uo pipefail
mkdir -p /tmp/s
echo {b64} | base64 -d > /tmp/s/test_initial_state.py
python3 -m pytest --version >/dev/null 2>&1 || python3 -m pip install -q pytest >/dev/null 2>&1 || true
python3 -m pytest -q -rA /tmp/s/test_initial_state.py > /tmp/s/out.txt 2>&1; rc=$?
echo "RC=$rc"
cat /tmp/s/out.txt
"""


def _tid(task):
    return task.metadata.get("metadata", {}).get("task_id") or task.id


def run_one(task, init_code):
    from rllm.eval._resolution import _create_sandbox_for_task
    from rllm.eval.script_evaluator import ShellScriptEvaluator

    rec = {"task_id": _tid(task), "ok": False, "error": None, "rc_init": None, "init_output": "", "reward": None, "final_output": ""}
    v_user = task.metadata.get("verifier_user")
    v_timeout = float(task.metadata.get("verifier_timeout") or 300.0)

    sb = None
    try:
        sb = _create_sandbox_for_task(task, "daytona")
        if init_code:
            out = sb.exec(_INIT.format(b64=base64.b64encode(init_code.encode()).decode()), timeout=v_timeout, user=v_user)
            rec["rc_init"] = int(out.split("RC=", 1)[1].split("\n", 1)[0])
            rec["init_output"] = out.split("\n", 1)[1] if "\n" in out else ""
        res = ShellScriptEvaluator(sb, verifier_user=v_user, verifier_timeout=v_timeout).evaluate(task, None)
        rec["reward"] = float(res.reward)
        rec["final_output"] = (res.metadata or {}).get("log_tail", "")
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        if sb is not None:
            try:
                sb.close()
            except Exception:
                pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-c", "--concurrency", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_dataset

    from rllm.tasks.loader import BenchmarkLoader

    tasks = list(BenchmarkLoader.load(str(DATASET_DIR), sandbox_backend="daytona").tasks)
    random.shuffle(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"loaded {len(tasks)} tasks", flush=True)

    print("loading test_initial_state from HF cache (offline)...", flush=True)
    init = {r["task_id"]: (r.get("test_initial_state") or "") for r in load_dataset(HF_REPO, split="train")}

    n = len(tasks)
    print(f"running {n} tasks, concurrency={args.concurrency}", flush=True)
    done = 0
    with open(args.output, "w") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_one, t, init.get(_tid(t))): _tid(t) for t in tasks}
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            if rec["ok"]:
                summary = f"OK  init_rc={rec['rc_init']} reward={rec['reward']}"
            else:
                summary = "ERR " + (rec["error"] or "")[:200]
            print(f"[{done}/{n}] {rec['task_id']}  {summary}", flush=True)

    print(f"done -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
