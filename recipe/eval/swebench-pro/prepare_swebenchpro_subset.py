#!/usr/bin/env python3
"""Build the SWE-bench Pro evaluation subset from the Harbor registry dataset.

Pulls ``harbor:swebenchpro`` if it is not registered yet, copies the task
directories listed in ``subset-ids.txt`` (one instance id per line; blank lines
and ``#`` comments ignored) into ``$RLLM_HOME/datasets/<name>/``,
applies the fixes below, and registers the rows so the subset runs under
either harness::

    rllm eval swebenchpro_100 --split test --agent harbor:mini-swe-agent --evaluator harbor_reward_fn ...
    rllm eval swebenchpro_100 --split test --agent mini-swe-agent ...

Fixes applied to the copies (the Harbor cache itself is left untouched):

* ``[environment] cpus`` / ``memory_mb`` -- the registry ships 1 CPU / 4 GiB.
  ``pytest -n auto`` (ansible), ``go test ./...`` (flipt, teleport, vuls,
  navidrome) and Jest (element-web, tutanota) size their parallelism by the
  host's nproc (256 here), so on 1 CPU they crawl past the verifier timeout
  and in 4 GiB the OOM killer takes out compilers and test workers
  (``signal: killed``, ``Killed``, pytest-xdist ``BrokenPipeError``). With
  4 CPUs / 16 GiB every task in the 100-task subset grades its gold patch;
  4 GiB was re-tested and fails 13 of them (2026-09-09).
* element-web ``tests/run_script.sh`` -- the registry adapter adds
  ``--maxWorkers=1 --forceExit`` to Jest (upstream SWE-bench_Pro-os never had
  it). In-band Jest lets ``@matrix-org/matrix-wysiwyg`` WASM state leak across
  test files and instance aec454dd then fails its own gold patch. The flags
  are the only difference from upstream, so they are removed.
* ``environment/Dockerfile`` -- appends an apt config that disables
  ``Check-Valid-Until``. Harbor's mini-swe-agent scaffold runs ``apt-get update``
  in the task container before installing itself; the Debian bullseye images
  (nodebb, element-web) fail that with ``Release file ... is expired`` since
  bullseye's archive metadata lapsed, so the agent never starts
  (``NonZeroAgentExitCodeError`` exit 100). The oracle harness is unaffected.
* ids: the registry lower-cases some instance ids (``instance_nodebb__...``);
  the id list is matched case-insensitively.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rllm import paths
from rllm.data import DatasetRegistry

HARBOR_DATASET = "swebenchpro"  # registered by `rllm dataset pull harbor:swebenchpro`
HARBOR_SPLIT = "default"
JEST_FLAGS = re.compile(r" --maxWorkers=1 --forceExit\b")


def read_ids(path: str) -> list[str]:
    ids = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return list(dict.fromkeys(ids))


def ensure_harbor_source():
    src = DatasetRegistry.load_dataset(HARBOR_DATASET, HARBOR_SPLIT)
    if src is None:
        print(f"'{HARBOR_DATASET}' not registered -- pulling harbor:{HARBOR_DATASET} ...", flush=True)
        subprocess.run(["rllm", "dataset", "pull", f"harbor:{HARBOR_DATASET}"], check=True)
        src = DatasetRegistry.load_dataset(HARBOR_DATASET, HARBOR_SPLIT)
    if src is None:
        sys.exit(f"could not load '{HARBOR_DATASET}/{HARBOR_SPLIT}' after pull")
    return src


def set_resources(task_toml: Path, cpus: int, memory_mb: int) -> None:
    text = task_toml.read_text()
    text = re.sub(r"^cpus\s*=.*$", f"cpus = {cpus}", text, flags=re.M)
    text = re.sub(r"^memory_mb\s*=.*$", f"memory_mb = {memory_mb}", text, flags=re.M)
    task_toml.write_text(text)


APT_NO_VALID_UNTIL = "RUN mkdir -p /etc/apt/apt.conf.d && echo 'Acquire::Check-Valid-Until \"false\";' > /etc/apt/apt.conf.d/99rllm-no-valid-until  # rllm: expired Debian Release files"


def patch_dockerfile(dockerfile: Path) -> bool:
    """Append the apt Check-Valid-Until override once (no-op on images without apt)."""
    text = dockerfile.read_text()
    if "99rllm-no-valid-until" in text:
        return False
    dockerfile.write_text(text.rstrip("\n") + "\n" + APT_NO_VALID_UNTIL + "\n")
    return True


def strip_jest_flags(run_script: Path) -> bool:
    text = run_script.read_text()
    new = JEST_FLAGS.sub("", text)
    if new != text:
        run_script.write_text(new)
        return True
    return False


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default=str(here / "subset-ids.txt"), help="Text file with one instance id per line")
    ap.add_argument("--name", default="swebenchpro_100")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--memory-mb", type=int, default=16384)
    ap.add_argument("--force", action="store_true", help="Re-copy task dirs that already exist")
    args = ap.parse_args()

    want = read_ids(args.ids)
    if not want:
        sys.exit(f"no instance ids in {args.ids}")
    src = ensure_harbor_source()
    by_id = {row["task_id"].lower(): row for row in src}

    out = Path(paths.rllm_path("datasets", args.name))
    out.mkdir(parents=True, exist_ok=True)
    rows, missing, patched = [], [], 0
    for iid in want:
        row = by_id.get(iid.lower())
        if row is None:
            missing.append(iid)
            continue
        src_dir = Path(row["task_path"])
        dst = out / src_dir.name
        if dst.exists() and args.force:
            shutil.rmtree(dst)
        if not dst.exists():
            shutil.copytree(src_dir, dst)
        set_resources(dst / "task.toml", args.cpus, args.memory_mb)
        if strip_jest_flags(dst / "tests" / "run_script.sh"):
            patched += 1
        patch_dockerfile(dst / "environment" / "Dockerfile")
        instruction = row.get("instruction") or (dst / "instruction.md").read_text()
        rows.append({"id": src_dir.name, "task_id": src_dir.name, "instruction": instruction, "question": instruction, "task_path": str(dst), "design_id": iid})

    if missing:
        sys.exit(f"{len(missing)} ids not found in '{HARBOR_DATASET}': {missing[:5]}")

    DatasetRegistry.register_dataset(
        name=args.name,
        data=rows,
        split=args.split,
        source=f"harbor:{HARBOR_DATASET} subset from {Path(args.ids).name}; cpus={args.cpus} memory_mb={args.memory_mb}; jest flags stripped",
        description=f"SWE-bench Pro subset ({len(rows)} tasks) from the Harbor registry with resource and run_script fixes",
        category="agentic",
    )
    print(f"{args.name}/{args.split}: {len(rows)} tasks -> {out}")
    print(f"  task.toml: cpus={args.cpus} memory_mb={args.memory_mb}; run_script.sh jest flags stripped in {patched} tasks; Dockerfile apt Check-Valid-Until off")
    print(f"  harbor harness: rllm eval {args.name} --split {args.split} --agent harbor:mini-swe-agent --evaluator harbor_reward_fn --sandbox-backend docker ...")
    print(f"  native harness: rllm eval {args.name} --split {args.split} --agent mini-swe-agent --sandbox-backend docker --agent-image auto ...")


if __name__ == "__main__":
    main()
