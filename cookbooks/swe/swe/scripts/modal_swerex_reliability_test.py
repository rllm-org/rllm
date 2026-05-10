#!/usr/bin/env python3
"""Concurrency/reliability probe for SWE-smith Modal/SWE-ReX calls.

This intentionally exercises the same stack used by SWEAgentFlow:
``swe.environment.create_env`` -> mini-swe-agent's ``swerex_modal``
environment -> SWE-ReX ``RemoteRuntime`` -> ``env.execute``.

The default "light" mode avoids a full SWE-smith setup while still touching the
same Modal create/execute/close endpoints.  The "setup" mode calls
``setup_swesmith_agent_env`` for a closer but heavier reproduction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rllm.data.dataset import DatasetRegistry  # noqa: E402
from swe.environment import create_env, ensure_bootstrapped  # noqa: E402
from swe.tasks.swesmith import setup_swesmith_agent_env  # noqa: E402
from swe.utils import close_env  # noqa: E402


TRANSIENT_RE = re.compile(
    r"Connection timeout to host|Response payload is not completed|ContentLengthError|"
    r"Cannot connect to host|Command timed out after|ServerDisconnectedError|Server disconnected|"
    r"ClientConnectorError|ConnectError|All connection attempts failed|ray client connection timeout",
    re.IGNORECASE,
)


LIGHT_COMMANDS: tuple[tuple[str, str, int], ...] = (
    ("pwd", "pwd", 30),
    ("python", "command -v python3 || command -v python", 30),
    ("git_status", "git status --short | head -20", 30),
    (
        "git_init",
        "rm -rf /tmp/modal_reliability_git && "
        "mkdir -p /tmp/modal_reliability_git && "
        "cd /tmp/modal_reliability_git && "
        "git init && "
        "git config user.email reliability@local && "
        "git config user.name reliability && "
        "touch probe.txt && "
        "git add probe.txt && "
        "git commit --allow-empty -m init",
        60,
    ),
)


def _now() -> float:
    return time.monotonic()


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "min": round(ordered[0], 3),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[p95_index], 3),
        "max": round(ordered[-1], 3),
    }


def _task_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    metadata = getattr(item, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    raise TypeError(f"Unsupported dataset item type: {type(item).__name__}")


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:1000]


def _run_light_commands(env: Any, task: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cwd = task["working_dir"]
    for name, command, timeout in LIGHT_COMMANDS:
        t0 = _now()
        result = env.execute({"command": command}, cwd=cwd, timeout=timeout)
        elapsed = _now() - t0
        output = result.get("output") or ""
        records.append(
            {
                "name": name,
                "elapsed_s": round(elapsed, 3),
                "returncode": result.get("returncode"),
                "output_preview": output[:300],
            }
        )
    return records


def run_one(
    *,
    idx: int,
    task: dict[str, Any],
    mode: str,
    command_timeout: int,
    sandbox_timeout: int,
) -> dict[str, Any]:
    sid = task.get("instance_id", f"idx-{idx}")
    record: dict[str, Any] = {
        "idx": idx,
        "instance_id": sid,
        "image": task.get("docker_image") or task.get("image_name"),
        "hostname": socket.gethostname(),
        "ok": False,
        "mode": mode,
        "timings": {},
        "commands": [],
        "error": None,
        "transient_error": False,
    }
    env = None
    total_start = _now()
    try:
        t0 = _now()
        env = create_env(task, command_timeout=command_timeout, sandbox_timeout=sandbox_timeout)
        record["timings"]["create_env_s"] = round(_now() - t0, 3)

        if mode == "setup":
            t0 = _now()
            setup_swesmith_agent_env(env, task, sid[:8], log_fn=lambda _msg: None)
            record["timings"]["setup_swesmith_s"] = round(_now() - t0, 3)
            record["ok"] = True
        else:
            record["commands"] = _run_light_commands(env, task)
            failed_commands = [cmd for cmd in record["commands"] if cmd.get("returncode") != 0]
            if failed_commands:
                joined = " | ".join(
                    f"{cmd['name']} rc={cmd.get('returncode')} {cmd.get('output_preview', '')[:160]}"
                    for cmd in failed_commands
                )
                record["error"] = f"command failure: {joined}"
                record["transient_error"] = bool(TRANSIENT_RE.search(joined))
            else:
                record["ok"] = True

        return record
    except Exception as exc:
        err = _short_error(exc)
        record["error"] = err
        record["transient_error"] = bool(TRANSIENT_RE.search(err) or TRANSIENT_RE.search(traceback.format_exc()))
        record["traceback_tail"] = traceback.format_exc()[-4000:]
        return record
    finally:
        if env is not None:
            t0 = _now()
            close_env(env)
            record["timings"]["close_env_s"] = round(_now() - t0, 3)
        record["timings"]["total_s"] = round(_now() - total_start, 3)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [r for r in records if not r["ok"]]
    transient = [r for r in failures if r.get("transient_error")]
    create_times = [r["timings"]["create_env_s"] for r in records if "create_env_s" in r["timings"]]
    total_times = [r["timings"]["total_s"] for r in records if "total_s" in r["timings"]]
    command_times: dict[str, list[float]] = {}
    for r in records:
        for cmd in r.get("commands", []):
            command_times.setdefault(cmd["name"], []).append(cmd["elapsed_s"])
    errors: dict[str, int] = {}
    for r in failures:
        key = (r.get("error") or "unknown").splitlines()[0][:180]
        errors[key] = errors.get(key, 0) + 1
    return {
        "total": len(records),
        "ok": len(records) - len(failures),
        "failed": len(failures),
        "transient_failed": len(transient),
        "success_rate": round((len(records) - len(failures)) / len(records), 4) if records else None,
        "create_env_s": _quantiles(create_times),
        "total_s": _quantiles(total_times),
        "command_s": {name: _quantiles(vals) for name, vals in sorted(command_times.items())},
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="swe_smith_filtered_mix")
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--mode", choices=("light", "setup"), default="light")
    parser.add_argument("--command-timeout", type=int, default=120)
    parser.add_argument("--sandbox-timeout", type=int, default=600)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    ensure_bootstrapped()
    dataset = DatasetRegistry.load_dataset(args.dataset)
    if dataset is None:
        raise SystemExit(f"Dataset not found: {args.dataset}")

    tasks = [_task_dict(dataset[i % len(dataset)]) for i in range(args.offset, args.offset + args.total)]
    out_path = Path(args.out) if args.out else Path(
        f"/tmp/modal_swerex_reliability_{socket.gethostname()}_{int(time.time())}.jsonl"
    )

    print(
        json.dumps(
            {
                "event": "start",
                "hostname": socket.gethostname(),
                "dataset": args.dataset,
                "total": args.total,
                "concurrency": args.concurrency,
                "mode": args.mode,
                "out": str(out_path),
                "swe_rex_remote_retries": os.getenv("SWE_REX_REMOTE_RETRIES"),
                "swe_rex_remote_sock_connect_timeout_s": os.getenv("SWE_REX_REMOTE_SOCK_CONNECT_TIMEOUT_S"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    records: list[dict[str, Any]] = []
    with out_path.open("w") as out, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_one,
                idx=i,
                task=task,
                mode=args.mode,
                command_timeout=args.command_timeout,
                sandbox_timeout=args.sandbox_timeout,
            )
            for i, task in enumerate(tasks)
        ]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            line = json.dumps(record, sort_keys=True)
            out.write(line + "\n")
            out.flush()
            print(line, flush=True)

    summary = summarize(records)
    print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
