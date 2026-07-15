"""Builder for the Scale-SWE sandbox benchmark.

Scale-SWE (``AweAI-Team/Scale-SWE``; arXiv:2602.09892 "Immersion in the
GitHub Universe") ships ~20K real-world Python bug-fix tasks as a flat HF
row dataset. Each row carries the problem statement plus a pre-built
per-instance Docker image (``aweaiteam/scaleswe:<instance_id>`` in
``image_url``) with the repo checked out under ``workdir``
(e.g. ``/workspace/<repo>``), the SWE-bench-style test fixtures
(``FAIL_TO_PASS`` / ``PASS_TO_PASS`` node ids), a synthetic reproduction
test (``f2p_script`` → ``test_fail_to_pass.py``), an optional test-infra
diff (``f2p_patch``), and the repo-setup shell (``pre_commands``) that
checks out the buggy ``parent_commit`` before the agent runs.

This builder expands each row into rLLM's sandbox (task-per-directory)
shape so ``rllm eval`` / ``AgentTrainer`` run each task through the
standard ``SandboxedAgentFlow`` + ``ShellScriptEvaluator`` path with no
new Python evaluator.

Verifier protocol mirrors the official BeyondSWE evaluator
(``AweAI-Team/AweAgent`` ``aweagent/tasks/beyond_swe/evaluator.py``):
``pre_commands`` runs *before* the agent (as a Dockerfile ``RUN`` step,
replayed by ``_replay_dockerfile`` on non-docker backends); the verifier
then applies ``f2p_patch`` on top of the agent's edits, writes
``f2p_script`` to ``<workdir>/test_fail_to_pass.py``, runs pytest over
``FAIL_TO_PASS ∪ PASS_TO_PASS``, and rewards 1.0 iff every required test
passes. Unlike SWE-bench Pro this does NOT reset-and-reapply the agent
diff — the official evaluator grades the agent's working tree directly.

On-disk output (``<out_dir>/``)::

    scaleswe/
    ├── dataset.toml                       # type="sandbox"
    ├── <instance_id>/
    │   ├── task.toml                      # docker_image=<image_url>, workdir=<workdir>
    │   ├── instruction.md                 # problem_statement
    │   ├── environment/Dockerfile         # FROM <image_url> + RUN <pre_commands>
    │   ├── tests/
    │   │   ├── test.sh                    # synthesized verifier (this module)
    │   │   └── instance.json             # workdir, f2p_patch, f2p_script, F2P/P2P
    │   └── solution/solve.sh              # apply the gold patch (oracle harness)
    └── ...

Invoked from ``rllm dataset pull scaleswe`` via the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_REPO_ID = "AweAI-Team/Scale-SWE"

# Per-instance resource defaults. Scale-SWE images bundle a full Python repo
# + its editable install; targeted pytest runs are light, but we match the
# other SWE builders so a remote backend's 1 GiB default doesn't OOM the
# grader. Docker ignores these; Modal/Daytona honor them (and the swe-rl
# cookbook clamps them down at runtime via RLLM_SANDBOX_MAX_*).
_DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory_mb": 16384,
    "storage_mb": 30720,
    "build_timeout_sec": 1800.0,
}

# Scale-SWE rows carry no per-task agent budget, so we don't invent one: the
# task.toml omits [agent].timeout_sec and the per-rollout wall-clock is governed
# by the harness run timeout (RLLM_HARNESS_RUN_TIMEOUT_S, default 3600) instead.
_DEFAULT_TIMEOUTS = {
    "verifier_timeout_sec": 1800.0,
}


def _is_test_path(path: str) -> bool:
    """Heuristic for a pytest test file (mirrors r2egym's FileDiff.is_test_file).

    Anything under a ``tests/``/``test/`` dir, or matching ``test_*.py`` /
    ``*_test.py`` / ``conftest.py``, is a test file.
    """
    if not path:
        return False
    last = path.split("/")[-1]
    if last == "conftest.py" or last.startswith("test_") or last.endswith("_test.py"):
        return True
    return any(p in {"tests", "Tests", "test", "Test"} for p in path.split("/"))


def _source_only_patch(patch: str) -> str:
    """Drop test-file sections from a unified-diff so only the source fix remains.

    Scale-SWE's ``patch`` is the *full* PR diff (source + tests). The oracle must
    apply only what an agent would change — the source — because the verifier
    separately applies ``f2p_patch`` to the (pristine) test files; keeping the
    PR's own test edits makes ``f2p_patch`` conflict. Splits on ``diff --git``
    headers and keeps sections whose path is not a test path. Falls back to the
    original text if there are no ``diff --git`` headers (nothing safe to split).
    """
    if not patch.strip() or "diff --git " not in patch:
        return patch
    sections: list[tuple[str, list[str]]] = []
    cur: tuple[str, list[str]] | None = None
    for ln in patch.splitlines(keepends=True):
        if ln.startswith("diff --git "):
            if cur is not None:
                sections.append(cur)
            parts = ln.split()
            path = ""
            if len(parts) >= 4:
                b = parts[3]
                path = b[2:] if b.startswith("b/") else b
            cur = (path, [ln])
        elif cur is not None:
            cur[1].append(ln)
    if cur is not None:
        sections.append(cur)
    kept = [ln for path, lines in sections if not _is_test_path(path) for ln in lines]
    return "".join(kept)


def _decode_json_list(value: Any) -> list[str]:
    """Parse a field that's either a JSON-encoded list or already a list.

    Scale-SWE stores ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` as JSON-string
    columns (all fields are typed ``string`` on HF), e.g. ``'["a", "b"]'``.
    Fall back to ``ast.literal_eval`` for Python-literal lists.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return [str(value)]
    text = value.strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import ast

            loaded = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            logger.warning("[scaleswe] could not parse list-valued field: %r", text[:120])
            return []
    if isinstance(loaded, list | tuple):
        return [str(v) for v in loaded]
    return [str(loaded)]


def _build_dockerfile(image_url: str, workdir: str, parent_commit: str) -> str:
    """``FROM <image_url>`` + clear ENTRYPOINT + a minimal checkout-to-baseline RUN.

    The agent must start at ``parent_commit`` (the buggy base the fix and the
    F2P/P2P tests target). The Scale-SWE images ship at a *different* HEAD, so a
    ``RUN`` step is needed to reach ``parent_commit`` before the agent:
      * docker: run at image build (``rllm eval`` builds this Dockerfile);
      * modal / daytona: replayed on the live sandbox by ``_replay_dockerfile``
        (``replay_dockerfile`` defaults true) before the agent's first turn.

    We deliberately do NOT replay the dataset's full ``pre_commands``. That
    script ends in ``git checkout -b scaleswe`` + ref pruning +
    ``git gc --prune=now --aggressive`` — pure image-shrinking hygiene. The
    branch step fails on the prebuilt image ("branch already exists"), and the
    aggressive ``gc`` is CPU-heavy: under the capped CPU + high build concurrency
    of training it dominates each snapshot build and makes it slow/flaky
    (spurious Modal ImageBuildFailed). ``git checkout -f <parent> && git reset
    --hard <parent>`` reaches the exact same tree with none of that cost, and
    doesn't touch untracked files (the repo's installed venv/egg-info survive).
    Wrapped in ``|| true`` so a benign failure never aborts ``docker build``.
    ENTRYPOINT is cleared so rLLM's ``sleep infinity`` keepalive wins.
    """
    lines = [f"FROM {image_url}", "ENTRYPOINT []"]
    wd = workdir or "/workspace"
    lines.append(f"WORKDIR {wd}")
    if parent_commit:
        # Run in the repo dir even though WORKDIR is set (replay drops WORKDIR).
        run = f'git checkout -f "{parent_commit}" && git reset --hard "{parent_commit}"'
        lines.append(f"RUN cd {wd} && ({run}) || true")
    return "\n".join(lines) + "\n"


def _build_task_toml(
    *,
    instance_id: str,
    repo: str,
    user: str,
    language: str,
    parent_commit: str,
    image_url: str,
    workdir: str,
) -> str:
    """Synthesize a Harbor-format ``task.toml``.

    The loader lifts ``[environment].docker_image`` / ``workdir`` / resources
    into ``task.metadata``. ``replay_dockerfile`` is set explicitly to ``true``:
    because an explicit ``docker_image`` is present, the loader would otherwise
    force ``replay_dockerfile=False`` (``_normalize_env_section``), skipping the
    Dockerfile's ``parent_commit`` checkout RUN on modal/daytona and grading
    against the image's (older) shipped HEAD.
    """
    wd = workdir or "/workspace"
    lang = language or "python"
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "scaleswe/{instance_id}"',
        f'description = "Scale-SWE: {user}/{repo} ({lang})"',
        f'keywords = ["scale-swe", "{lang}", "{repo}"]',
        "",
        "[metadata]",
        f'instance_id = "{instance_id}"',
        f'repo = "{repo}"',
        f'repo_user = "{user}"',
        f'language = "{lang}"',
        f'parent_commit = "{parent_commit}"',
        f'image_url = "{image_url}"',
        "",
        "[environment]",
        f'docker_image = "{image_url}"',
        # Explicit: with docker_image set, the loader defaults replay_dockerfile to
        # False; we need the Dockerfile's parent_commit checkout RUN to replay on
        # modal/daytona (docker runs it at build time).
        "replay_dockerfile = true",
        f'workdir = "{wd}"',
        f"cpus = {_DEFAULT_RESOURCES['cpus']}",
        f"memory_mb = {_DEFAULT_RESOURCES['memory_mb']}",
        f"storage_mb = {_DEFAULT_RESOURCES['storage_mb']}",
        f"build_timeout_sec = {_DEFAULT_RESOURCES['build_timeout_sec']}",
        "allow_internet = true",
        "",
        "[verifier]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['verifier_timeout_sec']}",
        "",
    ]
    return "\n".join(lines)


# Verifier template. Static across instances — per-task knobs (workdir,
# f2p_patch, f2p_script, F2P/P2P ids) live in tests/instance.json. Mirrors the
# official BeyondSWE evaluator (aweagent/tasks/beyond_swe/evaluator.py):
#
#   1. cd into the repo workdir (the agent's fix is already in the tree;
#      pre_commands ran at container entry via the Dockerfile RUN).
#   2. Apply f2p_patch (test-infra diff). Empty ⇒ skip; hard failure ⇒ 0.0
#      (matches the evaluator's "apply_patch failure → score 0.0").
#   3. Write f2p_script to <workdir>/test_fail_to_pass.py (the synthetic F2P
#      tests the FAIL_TO_PASS node ids reference).
#   4. Run pytest over FAIL_TO_PASS ∪ PASS_TO_PASS with -rA.
#   5. Reward 1.0 iff every required node id is PASSED in the summary; else 0.0.
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

INSTANCE_JSON=/tests/instance.json
[ -f "$INSTANCE_JSON" ] || { write_failure "tests/instance.json missing"; exit 0; }

WORKDIR="$(python3 -c "import json; print(json.load(open('$INSTANCE_JSON')).get('workdir') or '')")"
[ -n "$WORKDIR" ] || WORKDIR="$(pwd)"
cd "$WORKDIR" 2>/dev/null || { write_failure "workdir $WORKDIR missing"; exit 0; }
git config --global --add safe.directory "$WORKDIR" 2>/dev/null || true

# Step 1: apply f2p_patch on top of the agent's edits (test-infra changes the
# synthetic F2P tests need). Empty => skip. Try git apply, then plain patch.
python3 -c "import json; open('/tmp/f2p.patch','w').write(json.load(open('$INSTANCE_JSON')).get('f2p_patch') or '')"
if [ -s /tmp/f2p.patch ]; then
    if git apply -v /tmp/f2p.patch 2>&1 | tail -20; then
        log "f2p_patch applied via git apply"
    elif patch -p1 --forward --fuzz=3 < /tmp/f2p.patch >/dev/null 2>&1; then
        log "f2p_patch applied via patch -p1"
    else
        write_failure "f2p_patch failed to apply"
        exit 0
    fi
fi

# Step 2: write the synthetic reproduction tests to the repo root.
python3 -c "import json; open('test_fail_to_pass.py','w').write(json.load(open('$INSTANCE_JSON')).get('f2p_script') or '')"

# Step 3: gather the required node ids (F2P then P2P).
python3 - <<'PY'
import json
inst = json.load(open("/tests/instance.json"))
ids = list(inst.get("FAIL_TO_PASS") or []) + list(inst.get("PASS_TO_PASS") or [])
open("/tmp/test_ids.txt", "w").write("\n".join(ids))
PY
mapfile -t IDS < /tmp/test_ids.txt
if [ "${#IDS[@]}" -eq 0 ]; then
    write_failure "no FAIL_TO_PASS/PASS_TO_PASS tests declared"
    exit 0
fi

# Step 4: run pytest. Pick a python that can actually import pytest — the repo
# (and its test deps) live in the image's default python, which is usually the
# PATH ``python3`` (/usr/local/bin), NOT the distro ``/usr/bin/python3``.
PYBIN=""
for c in python3 python /usr/local/bin/python3 /usr/local/bin/python ./.venv/bin/python /opt/venv/bin/python /usr/bin/python3; do
    command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
    if "$c" -c "import pytest" >/dev/null 2>&1; then PYBIN="$c"; break; fi
done
: > /tmp/test_output.txt
if [ -n "$PYBIN" ]; then
    log "Running ${#IDS[@]} tests via $PYBIN -m pytest"
    "$PYBIN" -m pytest -rA -p no:cacheprovider "${IDS[@]}" > /tmp/test_output.txt 2>&1 || log "pytest exited non-zero (parser inspects log)"
elif command -v pytest >/dev/null 2>&1; then
    log "Running ${#IDS[@]} tests via pytest"
    pytest -rA -p no:cacheprovider "${IDS[@]}" > /tmp/test_output.txt 2>&1 || log "pytest exited non-zero (parser inspects log)"
else
    log "pytest not importable in any candidate python"
fi

# Step 5: score — reward 1.0 iff every required id PASSED.
python3 <<'PY'
import json, re
REWARD = "/tmp/rllm/reward.json"

def _tail(path, n=1500):
    try:
        return open(path).read()[-n:]
    except Exception:
        return ""

def decolor(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s or "")

inst = json.load(open("/tests/instance.json"))
f2p = list(inst.get("FAIL_TO_PASS") or [])
p2p = list(inst.get("PASS_TO_PASS") or [])
required = set(f2p) | set(p2p)

try:
    log = open("/tmp/test_output.txt").read()
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"reading test_output.txt: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

# pytest -rA prints a "short test summary info" footer with one line per test:
#   PASSED <nodeid>
#   FAILED <nodeid> - <reason>
# Parse it into a nodeid->status map; strip the " - <reason>" tail on failures.
section = log.split("short test summary info", 1)[1] if "short test summary info" in log else log
status = {}
for raw in section.splitlines():
    line = decolor(raw).strip()
    for st in ("PASSED", "FAILED", "ERROR"):
        if line.startswith(st + " "):
            nid = line[len(st):].strip().split(" - ")[0].strip()
            if nid:
                status[nid] = st
            break

passed = {nid for nid, st in status.items() if st == "PASSED"}
missing = sorted(required - passed)
matched = sorted(required & passed)
reward = 1.0 if required and not missing else 0.0

json.dump({
    "reward": reward,
    "is_correct": reward >= 1.0,
    "signals": {
        "f2p_required": len(f2p),
        "p2p_required": len(p2p),
        "passed_required": len(matched),
        "missing_required": len(missing),
    },
    "metadata": {
        "missing": missing[:50],
        "log_tail": _tail("/tmp/test_output.txt"),
    },
}, open(REWARD, "w"))
PY
"""


def _build_verifier_script() -> str:
    return _VERIFIER_TEMPLATE


def _build_solution_script(workdir: str, parent_commit: str, has_patch: bool) -> str:
    """``solution/solve.sh`` applies the gold patch — used by the ``oracle`` harness.

    Self-contained: it resets to ``parent_commit`` before applying the patch
    rather than trusting that ``pre_commands`` already did. The Scale-SWE images
    ship at a HEAD *older* than ``parent_commit`` (verified: apig-wsgi ships at
    #58, parent_commit is #79), and the gold ``patch`` is the PR fix generated
    against ``parent_commit`` — it only applies there. On the oracle path
    ``pre_commands`` may not have run (skipped replay / stale snapshot), so the
    repo can be at the image's default HEAD where ``git apply`` fails with
    "patch does not apply". A ``git checkout -f`` + ``git reset --hard`` to
    ``parent_commit`` (mirroring the SWE-bench Pro builder) makes the oracle
    robust regardless. Fail loudly if no patch was shipped.
    """
    wd = workdir or "/workspace"
    if not has_patch:
        return "#!/bin/bash\necho 'oracle solve.sh: no gold patch in row' >&2\nexit 1\n"
    reset = ""
    if parent_commit:
        reset = f'git checkout -f "{parent_commit}"\ngit reset --hard "{parent_commit}"\ngit clean -fd >/dev/null 2>&1 || true\n'
    return f"#!/bin/bash\nset -e\ncd {wd}\ngit config --global --add safe.directory {wd} 2>/dev/null || true\n{reset}git apply -v /solution/gold.patch || git apply --3way -v /solution/gold.patch\n"


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


def _extract_instruction(row: dict) -> str:
    """The curated GitHub-style issue is the agent prompt."""
    return (row.get("problem_statement") or "").strip() + "\n"


def _materialize_task(task_dir: Path, row: dict) -> dict:
    """Expand a single HF row into a Harbor-format task tree. Returns stats."""
    task_dir.mkdir(parents=True, exist_ok=True)

    instance_id = row["instance_id"]
    repo = row.get("repo") or ""
    user = row.get("user") or ""
    language = row.get("language") or "python"
    parent_commit = row.get("parent_commit") or ""
    image_url = row.get("image_url") or ""
    workdir = row.get("workdir") or "/workspace"

    (task_dir / "task.toml").write_text(
        _build_task_toml(
            instance_id=instance_id,
            repo=repo,
            user=user,
            language=language,
            parent_commit=parent_commit,
            image_url=image_url,
            workdir=workdir,
        ),
        encoding="utf-8",
    )

    (task_dir / "instruction.md").write_text(_extract_instruction(row), encoding="utf-8")

    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(image_url, workdir, parent_commit), encoding="utf-8")

    tests_dst = task_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    (tests_dst / "test.sh").write_text(_build_verifier_script(), encoding="utf-8")
    (tests_dst / "test.sh").chmod(0o755)
    instance_data = {
        "instance_id": instance_id,
        "workdir": workdir,
        "parent_commit": parent_commit,
        "f2p_patch": row.get("f2p_patch") or "",
        "f2p_script": row.get("f2p_script") or "",
        "FAIL_TO_PASS": _decode_json_list(row.get("FAIL_TO_PASS")),
        "PASS_TO_PASS": _decode_json_list(row.get("PASS_TO_PASS")),
    }
    (tests_dst / "instance.json").write_text(json.dumps(instance_data, indent=2), encoding="utf-8")

    sol_dst = task_dir / "solution"
    sol_dst.mkdir(parents=True, exist_ok=True)
    # Source-only: exclude the PR's own test-file edits so the verifier's
    # f2p_patch applies against pristine tests (see _source_only_patch).
    gold_patch = _source_only_patch(row.get("patch") or "")
    (sol_dst / "gold.patch").write_text(gold_patch, encoding="utf-8")
    (sol_dst / "solve.sh").write_text(_build_solution_script(workdir, parent_commit, bool(gold_patch.strip())), encoding="utf-8")
    (sol_dst / "solve.sh").chmod(0o755)

    return {
        "instance_id": instance_id,
        "f2p": len(instance_data["FAIL_TO_PASS"]),
        "p2p": len(instance_data["PASS_TO_PASS"]),
        "has_patch": bool(gold_patch.strip()),
    }


def _load_rows(hf_repo_id: str, hf_split: str, *, retries: int = 4, backoff_sec: float = 10.0) -> list[dict]:
    """Load HF rows with retries on transient Hub connection errors."""
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
            logger.warning("[scaleswe] load_dataset(%s) failed (attempt %d/%d): %s — retry in %.0fs", hf_repo_id, attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"load_dataset({hf_repo_id!r}, split={hf_split!r}) failed after {retries} attempts") from last_exc


def build_benchmark(
    *,
    name: str = "scaleswe",
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
    """Materialize Scale-SWE into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: Split label written into dataset.toml and the registry.
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``,
            ``default_agent``, ``source`` are read from it when present.
        task_ids: Build only these ``instance_id`` values.
        limit: Keep only the first N rows (after the ``task_ids`` filter).
        default_agent: ``default_agent`` written into dataset.toml.
        hf_repo_id: Override the HF dataset (default ``AweAI-Team/Scale-SWE``).
        hf_split: HF split to load (defaults to ``train`` — the only split).
        clean: Remove ``out_dir`` before building.
        register: Also register ``task_path`` rows in ``DatasetRegistry``.

    Returns:
        Path to the built benchmark directory.
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent
        hf_repo_id = hf_repo_id or catalog_entry.get("source")
    hf_repo_id = hf_repo_id or HF_REPO_ID

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[scaleswe] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[scaleswe] loading HF dataset %s split=%s ...", hf_repo_id, hf_split)
    rows = _load_rows(hf_repo_id, hf_split)

    if task_ids is not None:
        keep = set(task_ids)
        rows = [r for r in rows if r.get("instance_id") in keep]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[scaleswe] selected %d rows (task_ids=%s, limit=%s)", len(rows), task_ids and len(task_ids), limit)

    written = 0
    skipped = 0
    no_patch = 0
    for row in rows:
        instance_id = row.get("instance_id")
        if not instance_id:
            logger.warning("[scaleswe] row missing instance_id, skipping")
            skipped += 1
            continue
        if not row.get("image_url"):
            logger.warning("[scaleswe] %s: missing image_url, skipping", instance_id)
            skipped += 1
            continue
        task_dst = out / instance_id
        if task_dst.exists():
            shutil.rmtree(task_dst)
        stats = _materialize_task(task_dst, row)
        if not stats["has_patch"]:
            no_patch += 1
        written += 1

    description = (catalog_entry or {}).get("description") or (f"Scale-SWE ({hf_repo_id}): real-world Python SWE tasks with per-instance Docker images and SWE-bench-style F2P/P2P pytest grading.")
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)
    logger.info("[scaleswe] wrote %d task dirs to %s (skipped %d, no oracle patch %d)", written, out, skipped, no_patch)

    if register:
        try:
            from rllm.data import DatasetRegistry

            reg_rows = []
            for row in rows:
                iid = row.get("instance_id")
                if not iid:
                    continue
                task_dst = out / iid
                if not (task_dst / "task.toml").exists():
                    continue
                reg_rows.append(
                    {
                        "id": iid,
                        "instruction": (task_dst / "instruction.md").read_text(encoding="utf-8"),
                        "task_path": str(task_dst),
                        "repo": row.get("repo", ""),
                        "user": row.get("user", ""),
                        "image_url": row.get("image_url", ""),
                        "parent_commit": row.get("parent_commit", ""),
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
            logger.warning("[scaleswe] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out


def main() -> None:
    """CLI: ``python -m rllm.data.scaleswe_builder --out-dir <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize Scale-SWE into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="scaleswe")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-repo-id", default=None, help="Override HF source repo (default: AweAI-Team/Scale-SWE).")
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
