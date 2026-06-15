"""Builder for the R2E-Gym sandbox benchmark.

R2E-Gym (``R2E-Gym/R2E-Gym-Subset``, ``R2E-Gym/R2E-Gym-Lite``, etc.)
ships SWE-style tasks where each row carries a pre-built per-instance
Docker image (``namanjain12/<repo>_final:<commit>``) that already
contains the broken repo at ``/testbed``, the test fixtures at
``/r2e_tests/``, and the repo's own ``/testbed/run_tests.sh`` grader.
The agent fixes the bug; the verifier runs that ``run_tests.sh`` (after
exposing the fixtures inside the repo as ``/testbed/r2e_tests``) and
compares the parsed pytest output against ``expected_output_json``.

Schema is intentionally NOT SWE-bench-compatible — there's no
``instance_id`` / ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` / ``patch`` field.
The gold fix lives inside ``parsed_commit_content`` (a JSON serialization
of an R2E-Gym ``ParsedCommit`` pydantic model); we walk it ourselves to
emit the unified diff for the oracle harness.

Row schema (confirmed via HF API):
    repo_name, docker_image, commit_hash, parsed_commit_content,
    execution_result_content, expected_output_json, modified_files,
    relevant_files, prompt, problem_statement, num_non_test_*

On-disk output (``<out_dir>/``)::

    r2egym/
    ├── dataset.toml                       # type="sandbox"
    ├── <task_id>/                         # task_id = <repo>__<short_commit>
    │   ├── task.toml                      # docker_image=<row.docker_image>, workdir=/testbed
    │   ├── instruction.md                 # problem_statement (preferred) or prompt
    │   ├── environment/Dockerfile         # FROM <row.docker_image> + ENTRYPOINT []
    │   ├── tests/
    │   │   ├── test.sh                    # synthesized verifier
    │   │   └── instance.json              # expected_output_json + repo_name
    │   └── solution/solve.sh              # apply the gold patch
    └── ...

Invoked from ``rllm dataset pull r2egym`` via the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO_ID = "R2E-Gym/R2E-Gym-Subset"

# Per-instance resource defaults. R2E-Gym images bundle a uv venv + a
# full Python repo (orange3, numpy, pandas-style); tests can OOM with
# the 1 GiB remote-backend default. Docker ignores these.
_DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory_mb": 16384,
    "storage_mb": 30720,
    "build_timeout_sec": 1800.0,
}

_DEFAULT_TIMEOUTS = {
    "agent_timeout_sec": 1800.0,
    "verifier_timeout_sec": 1800.0,
}


def _is_test_path(path: str) -> bool:
    """Match R2E-Gym's FileDiff.is_test_file heuristic (commit_models/diff_classes.py).

    Treats anything under a ``tests/`` / ``Tests/`` / ``test/`` directory or
    matching the ``test_*.py`` / ``*_test.py`` filename convention as a test
    file. Used to strip test changes from the oracle patch — the test
    fixtures already live under ``/r2e_tests/`` in the docker image, so
    applying test-file diffs from the commit would double-stage them.
    """
    if path.endswith("_test.py"):
        return True
    parts = path.split("/")
    last = parts[-1] if parts else ""
    if last.startswith("test_"):
        return True
    return any(p in {"tests", "Tests", "test", "Test"} for p in parts)


def _emit_hunk(hunk: dict) -> list[str]:
    """Re-emit a single hunk as unified-diff text lines (no trailing newline)."""
    desc = hunk.get("descriptor") or {}
    old_r = desc.get("old_range") or {}
    new_r = desc.get("new_range") or {}
    section = desc.get("section") or ""
    section_suffix = f" {section}" if section else ""
    lines = [f"@@ -{old_r.get('start', 0)},{old_r.get('length', 0)} +{new_r.get('start', 0)},{new_r.get('length', 0)} @@{section_suffix}"]
    for line in (hunk.get("line_group") or {}).get("all_lines", []):
        t = line.get("type", "")
        c = line.get("content", "")
        if t == "context":
            lines.append(f" {c}")
        elif t == "added":
            lines.append(f"+{c}")
        elif t == "deleted":
            lines.append(f"-{c}")
        elif t == "note":
            lines.append(f"\\ {c}")
    return lines


def patch_from_parsed_commit(
    parsed_commit_json: str,
    *,
    include_test_files: bool = False,
    python_only: bool = True,
) -> str:
    """Reconstruct a unified ``git diff`` from R2E-Gym's serialized ParsedCommit.

    Walks ``file_diffs`` and re-emits the header / index_line / minus/plus /
    hunks the same way ``r2egym.commit_models.diff_classes.FileDiff.get_patch``
    does, so we don't take a runtime dependency on the heavyweight
    ``r2e-gym`` package (pulls in litellm, anthropic[vertex], torch,
    matplotlib, ...). Defaults match the oracle's needs: skip test files
    (already baked into the image at ``/r2e_tests/``) and skip non-Python
    files (.rst / .yml / etc. that the runtime never executes).
    """
    try:
        pc = json.loads(parsed_commit_json)
    except (TypeError, json.JSONDecodeError):
        return ""

    out: list[str] = []
    for fd in pc.get("file_diffs", []) or []:
        path = ((fd.get("header") or {}).get("file") or {}).get("path") or ""
        if not path:
            continue
        if not include_test_files and _is_test_path(path):
            continue
        if python_only and not path.endswith(".py"):
            continue

        # diff --git header (mirroring FileDiffHeader.get_patch)
        out.append(f"diff --git a/{path} b/{path}")
        misc = (fd.get("header") or {}).get("misc_line")
        if misc:
            out.append(str(misc))

        idx = fd.get("index_line") or {}
        if idx:
            old_hash = idx.get("old_commit_hash") or ""
            new_hash = idx.get("new_commit_hash") or ""
            mode = idx.get("mode") or ""
            if old_hash and new_hash:
                tail = f" {mode}" if mode else ""
                out.append(f"index {old_hash}..{new_hash}{tail}")

        if fd.get("is_binary_file"):
            bl = fd.get("binary_line")
            if bl:
                out.append(str(bl))

        m = fd.get("minus_file") or {}
        p = fd.get("plus_file") or {}
        if m.get("path") and p.get("path"):
            out.append(f"--- {m['path']}")
            out.append(f"+++ {p['path']}")

        for hunk in fd.get("hunks") or []:
            out.extend(_emit_hunk(hunk))

    return "\n".join(out) + ("\n" if out else "")


_SHORT_HASH_LEN = 12


def _task_id_for(row: dict) -> str:
    """Stable task id: ``<repo_name>__<short_commit_hash>``.

    R2E-Gym rows have no ``instance_id`` field; we synthesize one so the
    on-disk layout matches the other SWE-style benchmarks. Repo name has
    no slashes (it's just ``aiohttp`` / ``orange3`` / etc.), so the join
    is unambiguous.
    """
    repo = (row.get("repo_name") or "unknown").strip() or "unknown"
    commit = (row.get("commit_hash") or "")[:_SHORT_HASH_LEN]
    return f"{repo}__{commit}" if commit else repo


def _build_dockerfile(docker_image: str) -> str:
    """``FROM <docker_image>`` + clear ENTRYPOINT, same hazard as SWE-bench Pro."""
    return f"FROM {docker_image}\nENTRYPOINT []\nWORKDIR /testbed\n"


def _build_task_toml(
    *,
    task_id: str,
    repo: str,
    commit_hash: str,
    docker_image: str,
) -> str:
    """Synthesize a Harbor-format ``task.toml``."""
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "r2egym/{task_id}"',
        f'description = "R2E-Gym: {repo} @ {commit_hash[:_SHORT_HASH_LEN]}"',
        f'keywords = ["r2e-gym", "{repo}"]',
        "",
        "[metadata]",
        f'task_id = "{task_id}"',
        f'repo_name = "{repo}"',
        f'commit_hash = "{commit_hash}"',
        "",
        "[environment]",
        f'docker_image = "{docker_image}"',
        'workdir = "/testbed"',
        f"cpus = {_DEFAULT_RESOURCES['cpus']}",
        f"memory_mb = {_DEFAULT_RESOURCES['memory_mb']}",
        f"storage_mb = {_DEFAULT_RESOURCES['storage_mb']}",
        f"build_timeout_sec = {_DEFAULT_RESOURCES['build_timeout_sec']}",
        "allow_internet = true",
        "",
        "[agent]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['agent_timeout_sec']}",
        "",
        "[verifier]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['verifier_timeout_sec']}",
        "",
    ]
    return "\n".join(lines)


# The verifier is dead simple compared to swebench_pro: the
# R2E-Gym image ships a self-contained ``/testbed/run_tests.sh`` that runs
# the full test suite via pytest (``-W ignore -m pytest -rA r2e_tests``)
# from inside the repo's bundled ``.venv``. We expose the fixtures inside
# the repo (``/testbed/r2e_tests``), invoke the script (against whatever
# state the agent left /testbed in), parse the per-test PASSED/FAILED/ERROR
# statuses from the ``-rA`` short-summary footer, and compare to the
# expected map. Reward = 1.0 iff EVERY expected test appears with the
# expected status — matching ``_calculate_reward_r2e`` in
# ``r2egym.agenthub.runtime.docker``.
_VERIFIER_TEMPLATE = r"""#!/bin/bash
set -uo pipefail

mkdir -p /tmp/rllm /logs/verifier
REWARD_JSON=/tmp/rllm/reward.json

log() { echo "[verifier] $*"; }

write_failure() {
    python3 - "$1" <<'PY' || echo '{"reward": 0.0, "is_correct": false}' > "$REWARD_JSON"
import json, sys
json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": sys.argv[1]}}, open("/tmp/rllm/reward.json", "w"))
PY
}

cd /testbed 2>/dev/null || { write_failure "/testbed missing"; exit 0; }

# R2E-Gym's grader runs the repo's own ``run_tests.sh`` from the repo root.
# In the raw image that script lives at ``/testbed/run_tests.sh`` and invokes
# ``.venv/bin/python -W ignore -m pytest -rA r2e_tests`` — warnings are
# suppressed on purpose: the repo's setup.cfg turns warnings into errors, so
# without ``-W ignore`` unrelated tests flip to FAILED/ERROR and break the
# expected-output equality check. The test fixtures ship at ``/r2e_tests`` but
# the script references them as ``r2e_tests`` relative to /testbed, so they
# must be reachable inside the repo. r2egym does ``mv /r2e_tests
# /root/r2e_tests`` + a ``/testbed/r2e_tests`` symlink; we just make
# ``/testbed/r2e_tests`` resolve (symlink is non-destructive and collects
# identically — pytest's rootdir is /testbed either way).
if [ ! -e /testbed/r2e_tests ]; then
    if [ -d /r2e_tests ]; then
        ln -s /r2e_tests /testbed/r2e_tests
    elif [ -d /root/r2e_tests ]; then
        ln -s /root/r2e_tests /testbed/r2e_tests
    fi
fi

# Prefer the image's own run_tests.sh (it carries the correct -W ignore / -rA
# flags + the repo venv); fall back to the canonical command if it's absent.
RUN_TESTS=""
for cand in /testbed/run_tests.sh /root/run_tests.sh /r2e_tests/run_tests.sh; do
    if [ -f "$cand" ]; then
        RUN_TESTS="$cand"
        break
    fi
done
if [ -n "$RUN_TESTS" ]; then
    log "Running $RUN_TESTS"
    bash "$RUN_TESTS" > /tmp/test_output.txt 2>&1 || log "run_tests.sh exited non-zero (parser inspects log)"
elif [ -d /testbed/r2e_tests ]; then
    log "run_tests.sh not found; invoking pytest directly"
    PY=.venv/bin/python
    [ -x "$PY" ] || PY=python
    PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' "$PY" -W ignore -m pytest -rA r2e_tests > /tmp/test_output.txt 2>&1 || log "pytest exited non-zero (parser inspects log)"
else
    write_failure "no r2e_tests fixtures or run_tests.sh found"
    exit 0
fi

python3 <<'PY'
import json, re, sys
REWARD = "/tmp/rllm/reward.json"

def _tail(path, n=2000):
    try:
        return open(path).read()[-n:]
    except Exception:
        return ""

def decolor(s):
    return re.sub(r"\x1b\[\d+m", "", s or "")

# Mirror r2egym.repo_analysis.execution_log_parser.parse_log_pytest:
# tests are reported under the "short test summary info" footer that
# pytest emits with -ra/-rA. Each line starts with PASSED/FAILED/ERROR
# followed by the test id (path::Class::test_name); we collapse to the
# dotted-name form that expected_output_json uses
# (".".join(parts after the first :: split)).
def parse_log_pytest(log):
    status_map = {}
    if "short test summary info" not in log:
        return status_map
    section = log.split("short test summary info", 1)[1].strip()
    for raw in section.split("\n"):
        line = decolor(raw)
        if "PASSED" in line:
            name = ".".join(line.split("::")[1:])
            status_map[name] = "PASSED"
        elif "FAILED" in line:
            name = ".".join(line.split("::")[1:]).split(" - ")[0]
            status_map[name] = "FAILED"
        elif "ERROR" in line:
            try:
                name = ".".join(line.split("::")[1:])
            except IndexError:
                name = line
            name = name.split(" - ")[0]
            status_map[name] = "ERROR"
    return status_map

try:
    inst = json.load(open("/tests/instance.json"))
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"instance.json missing: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

try:
    expected = json.loads(inst.get("expected_output_json") or "{}")
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"expected_output_json malformed: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

# Drop ANSI color and strip pytest "<name> - <reason>" tails — the
# expected map sometimes carries the reason, sometimes doesn't.
expected = {decolor(k).split(" - ")[0]: v for k, v in expected.items()}

try:
    log = open("/tmp/test_output.txt").read()
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"reading test_output.txt: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

parsed = {decolor(k).split(" - ")[0]: v for k, v in parse_log_pytest(log).items()}

# Match r2egym's _calculate_reward_r2e: cardinalities must match AND
# every key must map to the same status. Either mismatch ⇒ reward 0.
if len(parsed) != len(expected):
    reward = 0.0
    mismatch = "size"
else:
    reward = 1.0
    mismatch = None
    for k, v in expected.items():
        if k not in parsed or parsed[k] != v:
            reward = 0.0
            mismatch = f"first_mismatch={k}"
            break

passed = sum(1 for v in parsed.values() if v == "PASSED")
expected_passed = sum(1 for v in expected.values() if v == "PASSED")
json.dump({
    "reward": reward,
    "is_correct": reward >= 1.0,
    "signals": {
        "expected_tests": len(expected),
        "parsed_tests": len(parsed),
        "passed_in_parsed": passed,
        "passed_in_expected": expected_passed,
    },
    "metadata": {
        "mismatch": mismatch,
        "log_tail": _tail("/tmp/test_output.txt", 1500),
    },
}, open(REWARD, "w"))
PY
"""


def _build_verifier_script() -> str:
    return _VERIFIER_TEMPLATE


def _build_solution_script(has_patch: bool) -> str:
    """Oracle harness: apply the reconstructed gold patch.

    We don't ``git reset --hard`` first — the image is
    already at the buggy state and the gold patch was reconstructed
    against that exact tree. Plain ``git apply`` of the patch reproduces
    the fix on top of /testbed. If no patch could be reconstructed
    (the ``parsed_commit_content`` JSON was empty or non-Python), fail
    loudly so the oracle eval surfaces the issue instead of silently
    "passing" against the unmodified buggy state.
    """
    if not has_patch:
        return "#!/bin/bash\necho 'oracle solve.sh: no gold patch reconstructable from parsed_commit_content' >&2\nexit 1\n"
    return "#!/bin/bash\nset -e\ncd /testbed\ngit config --global --add safe.directory /testbed 2>/dev/null || true\ngit apply -v /solution/gold.patch\n"


def _extract_instruction(row: dict) -> str:
    """Pick the cleanest task instruction.

    ``problem_statement`` is the curated GitHub-style issue (the canonical
    agent prompt). Some rows embed it inside ``[ISSUE]...[/ISSUE]`` tags
    (matches ``DockerRuntime.get_task_instruction`` in r2e-gym); strip
    those wrappers if present.
    """
    ps = (row.get("problem_statement") or "").strip()
    if ps:
        m = re.search(r"\[ISSUE\](.*?)\[/ISSUE\]", ps, re.DOTALL)
        if m:
            return m.group(1).strip() + "\n"
        return ps + "\n"
    return (row.get("prompt") or "").strip() + "\n"


def _write_dataset_toml(out: Path, *, name: str, split: str, description: str, default_agent: str) -> None:
    content = "\n".join(
        [
            "[dataset]",
            f'name = "{name}"',
            'type = "sandbox"',
            f'description = "{description}"',
            'default_sandbox = "docker"',
            f'default_agent = "{default_agent}"',
            f'split = "{split}"',
            "",
            "[verifier]",
            'script = "tests/test.sh"',
            "",
        ]
    )
    (out / "dataset.toml").write_text(content, encoding="utf-8")


def _materialize_task(task_dir: Path, row: dict) -> dict:
    """Expand a single HF row into a Harbor-format task tree. Returns stats."""
    task_dir.mkdir(parents=True, exist_ok=True)

    task_id = _task_id_for(row)
    repo = row.get("repo_name") or ""
    commit_hash = row.get("commit_hash") or ""
    docker_image = row.get("docker_image") or ""

    (task_dir / "task.toml").write_text(
        _build_task_toml(
            task_id=task_id,
            repo=repo,
            commit_hash=commit_hash,
            docker_image=docker_image,
        ),
        encoding="utf-8",
    )

    (task_dir / "instruction.md").write_text(_extract_instruction(row), encoding="utf-8")

    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(docker_image), encoding="utf-8")

    tests_dst = task_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    (tests_dst / "test.sh").write_text(_build_verifier_script(), encoding="utf-8")
    (tests_dst / "test.sh").chmod(0o755)
    instance_data = {
        "task_id": task_id,
        "repo_name": repo,
        "commit_hash": commit_hash,
        # expected_output_json is already a JSON string in the row; keep
        # it as a string so the verifier doesn't double-encode it.
        "expected_output_json": row.get("expected_output_json") or "{}",
    }
    (tests_dst / "instance.json").write_text(json.dumps(instance_data, indent=2), encoding="utf-8")

    sol_dst = task_dir / "solution"
    sol_dst.mkdir(parents=True, exist_ok=True)
    gold_patch = patch_from_parsed_commit(row.get("parsed_commit_content") or "")
    (sol_dst / "gold.patch").write_text(gold_patch, encoding="utf-8")
    (sol_dst / "solve.sh").write_text(_build_solution_script(bool(gold_patch.strip())), encoding="utf-8")
    (sol_dst / "solve.sh").chmod(0o755)

    return {"task_id": task_id, "has_patch": bool(gold_patch.strip())}


def _load_rows(hf_repo_id: str, hf_split: str, *, retries: int = 4, backoff_sec: float = 10.0) -> list[dict]:
    """Load HF rows with retries on transient Hub connection errors.

    ``rllm eval`` auto-pulls before running, so a transient
    ``LocalEntryNotFoundError`` from a DNS/SSL blip would crash the eval.
    Cached partial progress makes the retries cheap.
    """
    import time

    from datasets import load_dataset

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ds = load_dataset(hf_repo_id, split=hf_split)
            return [dict(r) for r in ds]
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            wait = backoff_sec * attempt
            logger.warning("[r2egym] load_dataset(%s) failed (attempt %d/%d): %s — retry in %.0fs", hf_repo_id, attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"load_dataset({hf_repo_id!r}, split={hf_split!r}) failed after {retries} attempts") from last_exc


def build_benchmark(
    *,
    name: str = "r2egym",
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "mini-swe-agent",
    hf_repo_id: str | None = None,
    hf_split: str = "train",
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize R2E-Gym into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: Split label written into dataset.toml and the registry.
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``,
            ``default_agent``, ``source`` are read from it when present.
        task_ids: Build only these task ids (``<repo>__<short_commit>``).
        limit: Keep only the first N rows (after the ``task_ids`` filter).
        default_agent: ``default_agent`` written into dataset.toml.
        hf_repo_id: Override the HF dataset. Defaults to the catalog
            ``source`` or ``R2E-Gym/R2E-Gym-Subset`` (4,578 train rows).
            Use ``R2E-Gym/R2E-Gym-Lite`` for the 11K-row version with
            multiple dev splits.
        hf_split: HF split to load (``train`` for Subset; Lite ships
            ``dev_*`` splits too).
        clean: Remove ``out_dir`` before building.
        register: Also register ``task_path`` rows in ``DatasetRegistry``.

    Returns:
        Path to the built benchmark directory.
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent
        hf_repo_id = hf_repo_id or catalog_entry.get("source")
    hf_repo_id = hf_repo_id or DEFAULT_HF_REPO_ID

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[r2egym] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[r2egym] loading HF dataset %s split=%s ...", hf_repo_id, hf_split)
    rows = _load_rows(hf_repo_id, hf_split)

    if task_ids is not None:
        keep = set(task_ids)
        rows = [r for r in rows if _task_id_for(r) in keep]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[r2egym] selected %d rows (task_ids=%s, limit=%s)", len(rows), task_ids and len(task_ids), limit)

    written = 0
    skipped = 0
    no_patch = 0
    for row in rows:
        task_id = _task_id_for(row)
        if not row.get("docker_image"):
            logger.warning("[r2egym] %s: missing docker_image, skipping", task_id)
            skipped += 1
            continue
        task_dst = out / task_id
        if task_dst.exists():
            shutil.rmtree(task_dst)
        stats = _materialize_task(task_dst, row)
        if not stats["has_patch"]:
            no_patch += 1
        written += 1

    description = (catalog_entry or {}).get("description") or (
        f"R2E-Gym ({hf_repo_id}): real-world Python SWE tasks with per-instance Docker images and pytest-output equality grading against expected outputs."
    )
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)
    logger.info("[r2egym] wrote %d task dirs to %s (skipped %d, no oracle patch %d)", written, out, skipped, no_patch)

    if register:
        try:
            from rllm.data import DatasetRegistry

            reg_rows = []
            for row in rows:
                task_id = _task_id_for(row)
                task_dst = out / task_id
                if not (task_dst / "task.toml").exists():
                    continue
                reg_rows.append(
                    {
                        "id": task_id,
                        "instruction": (task_dst / "instruction.md").read_text(encoding="utf-8"),
                        "task_path": str(task_dst),
                        "repo_name": row.get("repo_name", ""),
                        "commit_hash": row.get("commit_hash", ""),
                        "docker_image": row.get("docker_image", ""),
                    }
                )
            DatasetRegistry.register_dataset(
                name=name,
                data=reg_rows,
                split=split,
                source=hf_repo_id,
                description=description,
                category=(catalog_entry or {}).get("category", "code"),
            )
        except Exception:
            logger.warning("[r2egym] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out


def main() -> None:
    """CLI: ``python -m rllm.data.r2egym_builder --out-dir <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize R2E-Gym into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="r2egym")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-repo-id", default=None, help="Override HF source repo (default: R2E-Gym/R2E-Gym-Subset).")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--default-agent", default="mini-swe-agent")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("RLLM_LOG_LEVEL", "INFO"))
    build_benchmark(
        name=args.name,
        split=args.split,
        out_dir=args.out_dir,
        task_ids=args.task_ids,
        limit=args.limit,
        default_agent=args.default_agent,
        hf_repo_id=args.hf_repo_id,
        hf_split=args.hf_split,
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()
