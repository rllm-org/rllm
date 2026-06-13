"""Builder for the SWE-bench Pro sandbox benchmark.

SWE-bench Pro (``ScaleAI/SWE-bench_Pro``) ships as a flat HF row dataset:
each row carries the problem statement plus the per-instance Docker tag
(``dockerhub_tag``), test fixtures (``fail_to_pass`` / ``pass_to_pass``),
the test file allowlist (``selected_test_files_to_run``), and a
``before_repo_set_cmd`` whose last line typically checks out the gold
test files from a later commit.

The per-instance ``run_script.sh`` + ``parser.py`` (one of each per task,
~1500 small files in total) live in the companion repo at
``github.com/scaleapi/SWE-bench_Pro-os`` under ``run_scripts/<instance_id>/``.
This builder downloads both, then expands each row into rLLM's sandbox
(task-per-directory) shape so ``rllm eval`` runs each task through the
standard ``SandboxedAgentFlow`` + ``ShellScriptEvaluator`` path with no
new Python evaluator needed.

On-disk output (``<out_dir>/``)::

    swebench_pro/
    ├── dataset.toml                       # type="sandbox"
    ├── <instance_id>/
    │   ├── task.toml                      # docker_image=jefzda/sweap-images:<tag>, workdir=/app
    │   ├── instruction.md                 # problem_statement [+ requirements / interface]
    │   ├── environment/Dockerfile         # FROM jefzda/sweap-images:<tag>
    │   ├── tests/
    │   │   ├── test.sh                    # synthesized verifier (this module)
    │   │   ├── run_script.sh              # per-instance, copied from SWE-bench_Pro-os
    │   │   ├── parser.py                  # per-instance, copied from SWE-bench_Pro-os
    │   │   └── instance.json              # base_commit, before_repo_set_cmd, selected/F2P/P2P
    │   └── solution/solve.sh              # apply the gold patch
    └── ...

The verifier replicates the upstream ``swe_bench_pro_eval.py`` flow:
capture the agent's ``git diff`` at ``/app``, hard-reset to base_commit,
re-apply the diff, run the last line of ``before_repo_set_cmd`` (which
brings in the hidden test files), invoke ``run_script.sh``, parse
results via ``parser.py``, and reward 1.0 iff every name in
``fail_to_pass ∪ pass_to_pass`` is in the parser's PASSED set.

Invoked from ``rllm dataset pull swebench_pro`` via the ``builder`` field
in ``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_REPO_ID = "ScaleAI/SWE-bench_Pro"
SCRIPTS_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
DOCKERHUB_NAMESPACE = "jefzda/sweap-images"

# Per-instance resource defaults. SWE-bench Pro instances ship full JS / Go /
# Python test suites — the upstream eval allocates 1–4 CPU and 5–30 GiB. We
# err on the higher end so the in-sandbox test runner (Mocha/pytest/go test)
# doesn't OOM. Docker ignores these; Modal/Daytona honor them.
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


def _shallow_clone_scripts_repo() -> Path:
    """Shallow-clone the SWE-bench_Pro-os repo into a temp dir.

    Only ``run_scripts/`` is consumed (1000+ tiny shell + python files); a
    full clone would pull traj/, dockerfiles/, etc. that we don't need. Git
    sparse-checkout works but adds complexity; a plain shallow clone is
    ~50 MB and one-shot.
    """
    tmp = Path(tempfile.mkdtemp(prefix="swebench-pro-scripts-"))
    logger.info("[swebench_pro] cloning %s into %s ...", SCRIPTS_REPO_URL, tmp)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", SCRIPTS_REPO_URL, str(tmp)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        out = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        raise RuntimeError(f"git clone of {SCRIPTS_REPO_URL} failed:\n{out}") from e
    return tmp


def _decode_json_list(value: Any) -> list[str]:
    """Parse a field that's either a JSON-encoded list or already a list.

    SWE-bench Pro stores ``fail_to_pass`` / ``pass_to_pass`` /
    ``selected_test_files_to_run`` as Python-literal-list strings on HF
    (e.g. ``'["a", "b"]'``). The upstream eval ``eval()`` s them; we
    ``json.loads`` after substituting single quotes when needed.
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
            logger.warning("[swebench_pro] could not parse list-valued field: %r", text[:120])
            return []
    return [str(v) for v in (loaded or [])]


def _last_nonblank_line(text: str) -> str:
    """Mirror upstream's ``before_repo_set_cmd.strip().split('\\n')[-1]`` semantics."""
    if not text:
        return ""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _scripts_dir_for(scripts_root: Path, instance_id: str) -> Path | None:
    """Resolve ``<scripts_root>/run_scripts/<instance_id>/`` with one fallback.

    The HF ``instance_id`` already includes the ``instance_`` prefix
    (e.g. ``instance_NodeBB__NodeBB-...-vnan``), matching the on-disk
    directory name. Some older mirrors strip the prefix — try both.
    """
    base = scripts_root / "run_scripts"
    primary = base / instance_id
    if primary.is_dir():
        return primary
    if instance_id.startswith("instance_"):
        alt = base / instance_id[len("instance_") :]
    else:
        alt = base / f"instance_{instance_id}"
    if alt.is_dir():
        return alt
    return None


def _build_instruction(row: dict) -> str:
    """Compose the agent instruction from problem_statement + extras.

    SWE-bench Pro splits the brief into three fields:
    - ``problem_statement``: the issue / bug report
    - ``requirements``: explicit deliverables (sometimes null)
    - ``interface``: API / interface spec the fix must conform to
    Both extras are optional. We surface them as labeled sections so the
    agent gets the same context the upstream eval gives.
    """
    parts: list[str] = []
    problem = (row.get("problem_statement") or "").strip()
    if problem:
        parts.append(problem)
    requirements = (row.get("requirements") or "").strip()
    if requirements:
        parts.append("## Requirements\n\n" + requirements)
    interface = (row.get("interface") or "").strip()
    if interface:
        parts.append("## Interface\n\n" + interface)
    return "\n\n".join(parts).rstrip() + "\n"


def _build_dockerfile(dockerhub_tag: str) -> str:
    """Dockerfile that pulls jefzda/sweap-images:<tag> and clears the base ENTRYPOINT.

    The image already ships with the repo at ``/app`` checked out to
    ``base_commit``, plus the test-runner dependencies (npm/pytest/go).

    The base SWE-bench Pro images declare ``ENTRYPOINT ["/bin/bash"]``.
    rLLM's docker backend (``rllm/sandbox/backends/docker.py``) starts
    containers with ``command="sleep infinity"`` to keep them alive for
    the agent + verifier execs — but the inherited entrypoint turns the
    invocation into ``/bin/bash sleep infinity``, which bash interprets
    as ``read script file 'sleep'`` and exits immediately. The container
    dies before the first exec, then every ``sandbox.exec`` returns
    HTTP 409 ``container is not running``. Reset ENTRYPOINT to ``[]``
    so docker uses our ``sleep infinity`` command directly.
    """
    return f"FROM {DOCKERHUB_NAMESPACE}:{dockerhub_tag}\nENTRYPOINT []\nWORKDIR /app\n"


def _build_task_toml(
    *,
    instance_id: str,
    repo: str,
    repo_language: str,
    base_commit: str,
    dockerhub_tag: str,
) -> str:
    """Synthesize a Harbor-format ``task.toml``.

    The loader lifts ``[environment].docker_image`` / ``workdir`` /
    resources into ``task.metadata`` so non-docker backends pull the image
    rather than rebuilding the Dockerfile.
    """
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "swebench_pro/{instance_id}"',
        f'description = "SWE-bench Pro: {repo} ({repo_language})"',
        f'keywords = ["swe-bench-pro", "{repo_language}"]',
        "",
        "[metadata]",
        f'instance_id = "{instance_id}"',
        f'repo = "{repo}"',
        f'repo_language = "{repo_language}"',
        f'base_commit = "{base_commit}"',
        f'dockerhub_tag = "{dockerhub_tag}"',
        "",
        "[environment]",
        f'docker_image = "{DOCKERHUB_NAMESPACE}:{dockerhub_tag}"',
        'workdir = "/app"',
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


# Verifier script template. Uses python3 (always present in sweap-images;
# jq is not) for JSON parsing and reward computation. Logic mirrors the
# upstream entryscript in ``swe_bench_pro_eval.py:create_entryscript``:
#
#   1. ``git diff`` at /app captures the agent's edits as a patch (binary
#      hunks included so png/icon edits don't break apply).
#   2. ``git reset --hard {base_commit}`` + ``git checkout {base_commit}``
#      reset the worktree to the pre-fix baseline.
#   3. ``git apply`` re-applies the agent's patch on top of the reset.
#   4. The last non-blank line of ``before_repo_set_cmd`` runs (matches
#      upstream's ``.split('\n')[-1]`` semantics) — typically a
#      ``git checkout <future-commit> -- <test files>`` that brings in
#      the hidden test fixtures.
#   5. ``run_script.sh`` runs with the selected test files; ``parser.py``
#      normalizes its output into ``{"tests": [{"name", "status"}]}``.
#   6. Reward is 1.0 iff every ``fail_to_pass ∪ pass_to_pass`` name is in
#      the parser's PASSED set; else 0.0.
_VERIFIER_TEMPLATE = r"""#!/bin/bash
set -uo pipefail

mkdir -p /tmp/rllm /logs/verifier
REWARD_JSON=/tmp/rllm/reward.json

log() { echo "[verifier] $*"; }

write_failure() {
    local reason="$1"
    python3 - "$reason" <<'PY' || echo '{"reward": 0.0, "is_correct": false}' > "$REWARD_JSON"
import json, sys
json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": sys.argv[1]}}, open("/tmp/rllm/reward.json", "w"))
PY
}

cd /app 2>/dev/null || { write_failure "/app missing"; exit 0; }

git config --global --add safe.directory "$(pwd)" 2>/dev/null || true

INSTANCE_JSON=/tests/instance.json
if [ ! -f "$INSTANCE_JSON" ]; then
    write_failure "tests/instance.json missing"
    exit 0
fi

BASE_COMMIT="$(python3 -c "import json; print(json.load(open('$INSTANCE_JSON'))['base_commit'])")"
if [ -z "$BASE_COMMIT" ]; then
    write_failure "base_commit empty"
    exit 0
fi

# Step 1: capture the agent's diff vs HEAD (the image checks out base_commit
# at build time, so HEAD ≡ base_commit when the agent starts).
log "Capturing agent diff vs HEAD ($BASE_COMMIT)"
MODEL_PATCH=/tmp/model_patch.diff
git add -A . >/dev/null 2>&1 || true
git diff --cached --binary > "$MODEL_PATCH" 2>/dev/null || true
git reset >/dev/null 2>&1 || true

PATCH_BYTES=$(wc -c < "$MODEL_PATCH" 2>/dev/null || echo 0)
log "Captured patch: $PATCH_BYTES bytes"

# Step 2: reset to base_commit so the patch applies on a clean tree.
log "Resetting to base_commit"
git reset --hard "$BASE_COMMIT" >/dev/null 2>&1 || log "git reset --hard failed (continuing)"
git checkout "$BASE_COMMIT" >/dev/null 2>&1 || log "git checkout failed (continuing)"

# Step 3: re-apply the agent's patch.
if [ -s "$MODEL_PATCH" ]; then
    if ! git apply -v "$MODEL_PATCH" 2>&1 | tail -20; then
        log "git apply failed — tests will run against unmodified base_commit"
    fi
else
    log "No agent changes detected"
fi

# Step 4: run the last non-blank line of before_repo_set_cmd. Upstream
# eval (swe_bench_pro_eval.py:create_entryscript) does the same — this is
# typically a `git checkout <future-commit> -- <test files>` that pulls in
# the hidden test fixtures the run_script.sh will exercise.
BEFORE_CMD="$(python3 -c "
import json
data = json.load(open('$INSTANCE_JSON'))
cmd = (data.get('before_repo_set_cmd') or '').strip().splitlines()
print(cmd[-1].strip() if cmd else '', end='')
")"
if [ -n "$BEFORE_CMD" ]; then
    log "before_repo_set_cmd: $BEFORE_CMD"
    bash -c "$BEFORE_CMD" || log "before_repo_set_cmd exited non-zero (continuing)"
fi

# Step 5: run the per-instance run_script.sh with the selected test files.
SELECTED="$(python3 -c "
import json
data = json.load(open('$INSTANCE_JSON'))
print(','.join(data.get('selected_test_files_to_run') or []), end='')
")"
log "Running tests: $SELECTED"
chmod +x /tests/run_script.sh
bash /tests/run_script.sh "$SELECTED" > /tmp/stdout.log 2> /tmp/stderr.log || log "run_script.sh exited non-zero (parser inspects logs)"

# Step 6: parse the run_script.sh output into the standard JSON shape.
python3 /tests/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json 2>>/tmp/stderr.log || log "parser.py failed"

# Step 7: compare passed set against fail_to_pass ∪ pass_to_pass.
python3 <<'PY'
import json, os
REWARD = "/tmp/rllm/reward.json"
try:
    inst = json.load(open("/tests/instance.json"))
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"instance.json missing: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

def _tail(path, n=2000):
    try:
        return open(path).read()[-n:]
    except Exception:
        return ""

try:
    out = json.load(open("/tmp/output.json"))
except Exception as e:
    json.dump({
        "reward": 0.0, "is_correct": False,
        "metadata": {"error": f"parser produced no output.json: {e}",
                     "stdout_tail": _tail("/tmp/stdout.log"),
                     "stderr_tail": _tail("/tmp/stderr.log")}
    }, open(REWARD, "w"))
    raise SystemExit(0)

f2p = set(inst.get("fail_to_pass") or [])
p2p = set(inst.get("pass_to_pass") or [])
required = f2p | p2p

passed = {t.get("name", "") for t in (out.get("tests") or []) if t.get("status") == "PASSED"}
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
        "passed_required_sample": matched[:10],
    },
}, open(REWARD, "w"))
PY
"""


def _build_verifier_script() -> str:
    """Return ``tests/test.sh`` content (identical across all instances).

    The per-task knobs (base_commit, F2P/P2P, before_repo_set_cmd,
    selected files) live in ``tests/instance.json`` so the verifier
    script itself stays static — easier to audit and to patch in a
    single place if the upstream eval flow changes.
    """
    return _VERIFIER_TEMPLATE


def _build_solution_script(patch: str) -> str:
    """``solution/solve.sh`` applies the gold patch — used by the ``oracle`` harness.

    Mirrors :func:`swesmith_builder.build_benchmark`'s solution layout.
    """
    return "#!/bin/bash\nset -e\ncd /app\ngit config --global --add safe.directory /app 2>/dev/null || true\ngit apply -v /solution/gold.patch\n"


def _patch_existing_dockerfiles(out: Path) -> int:
    """Backfill ``ENTRYPOINT []`` into Dockerfiles written by older revisions.

    The first sandbox-builder revision emitted a 2-line Dockerfile (just
    ``FROM`` + ``WORKDIR``). Without ``ENTRYPOINT []`` the inherited base
    entrypoint (``["/bin/bash"]``) eats rLLM's ``sleep infinity`` start
    command and the container exits immediately. Re-walk existing task
    dirs and inject the missing line so a re-pull isn't required —
    callers that already have ~/.rllm/datasets/swebench_pro/<id>/ from
    a prior pull pick up the fix on the next build. New tasks are
    written with the corrected line directly by :func:`_build_dockerfile`.
    """
    patched = 0
    for task_dir in out.iterdir():
        if not task_dir.is_dir():
            continue
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            continue
        text = dockerfile.read_text(encoding="utf-8")
        if "ENTRYPOINT" in text:
            continue
        # Inject after the first FROM line. Plain string ops — these
        # Dockerfiles are 2 lines today; no need for a parser.
        lines = text.splitlines()
        new_lines: list[str] = []
        injected = False
        for line in lines:
            new_lines.append(line)
            if not injected and line.lstrip().upper().startswith("FROM "):
                new_lines.append("ENTRYPOINT []")
                injected = True
        if injected:
            dockerfile.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
            patched += 1
    return patched


def _purge_row_materialize_artifacts(out: Path) -> None:
    """Remove artifacts written by the HF-row materialize path.

    ``swebench_pro`` shipped briefly as an HF-row dataset (a previous
    revision of this PR). Its first ``rllm dataset pull`` landed:

        ~/.rllm/datasets/swebench_pro/
        ├── dataset.toml          # transform-style: no [verifier]
        ├── data/test.jsonl       # the row dump
        └── instruction.md.tpl    # row→prompt template

    A subsequent ``pull`` with the sandbox builder writes the new
    ``dataset.toml`` + ``<instance_id>/`` task tree alongside them — but
    leaves ``data/`` in place. ``BenchmarkLoader._has_data_file`` then
    sees ``data/`` and routes through ``_load_data_dataset`` (one Task
    per row, ``sub_dir=None``, ``task.id`` numeric) instead of
    ``_load_sandbox_dataset``. Symptom: every task fails with
    ``missing reference solution at <dataset_root>/solution/solve.sh``
    because ``task.task_dir`` == dataset root, not the instance dir.

    Wipe the conflicting paths up front so re-pulls converge regardless
    of the previous shape this dataset was pulled in.
    """
    for stale in ("data", "images", "instruction.md.tpl"):
        target = out / stale
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            try:
                target.unlink()
            except OSError:
                pass


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


def _materialize_task(
    task_dir: Path,
    row: dict,
    scripts_dir: Path,
) -> dict:
    """Expand a single HF row into a Harbor-format task tree. Returns stats."""
    task_dir.mkdir(parents=True, exist_ok=True)

    instance_id = row["instance_id"]
    repo = row.get("repo", "")
    repo_language = row.get("repo_language", "")
    base_commit = row.get("base_commit", "")
    dockerhub_tag = row.get("dockerhub_tag", "")

    # task.toml
    (task_dir / "task.toml").write_text(
        _build_task_toml(
            instance_id=instance_id,
            repo=repo,
            repo_language=repo_language,
            base_commit=base_commit,
            dockerhub_tag=dockerhub_tag,
        ),
        encoding="utf-8",
    )

    # instruction.md
    (task_dir / "instruction.md").write_text(_build_instruction(row), encoding="utf-8")

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(dockerhub_tag), encoding="utf-8")

    # tests/
    tests_dst = task_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    (tests_dst / "test.sh").write_text(_build_verifier_script(), encoding="utf-8")
    (tests_dst / "test.sh").chmod(0o755)

    # Copy per-instance run_script.sh + parser.py.
    src_run_script = scripts_dir / "run_script.sh"
    src_parser = scripts_dir / "parser.py"
    shutil.copy2(src_run_script, tests_dst / "run_script.sh")
    (tests_dst / "run_script.sh").chmod(0o755)
    shutil.copy2(src_parser, tests_dst / "parser.py")

    instance_data = {
        "instance_id": instance_id,
        "base_commit": base_commit,
        "before_repo_set_cmd": row.get("before_repo_set_cmd", "") or "",
        "selected_test_files_to_run": _decode_json_list(row.get("selected_test_files_to_run")),
        "fail_to_pass": _decode_json_list(row.get("fail_to_pass")),
        "pass_to_pass": _decode_json_list(row.get("pass_to_pass")),
        "repo": repo,
        "repo_language": repo_language,
    }
    (tests_dst / "instance.json").write_text(json.dumps(instance_data, indent=2), encoding="utf-8")

    # solution/ (gold patch + apply script for the oracle harness)
    sol_dst = task_dir / "solution"
    sol_dst.mkdir(parents=True, exist_ok=True)
    patch = row.get("patch") or ""
    (sol_dst / "gold.patch").write_text(patch, encoding="utf-8")
    (sol_dst / "solve.sh").write_text(_build_solution_script(patch), encoding="utf-8")
    (sol_dst / "solve.sh").chmod(0o755)

    return {"f2p": len(instance_data["fail_to_pass"]), "p2p": len(instance_data["pass_to_pass"])}


def _load_rows(hf_split: str) -> list[dict]:
    """Load the HF dataset rows as a list of plain dicts."""
    from datasets import load_dataset

    ds = load_dataset(HF_REPO_ID, split=hf_split)
    return [dict(r) for r in ds]


def build_benchmark(
    *,
    name: str = "swebench_pro",
    split: str = "test",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "mini-swe-agent",
    hf_split: str = "test",
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize SWE-bench Pro into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: Split label written into dataset.toml and the registry.
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``
            and ``default_agent`` are read from it when present.
        task_ids: Build only these ``instance_id`` values. Default: all 731.
        limit: Keep only the first N rows (after the ``task_ids`` filter).
        default_agent: ``default_agent`` written into dataset.toml.
        hf_split: HF split to load (defaults to ``test`` — the only split
            in ``ScaleAI/SWE-bench_Pro``).
        clean: Remove ``out_dir`` before building.
        register: Also register ``task_path`` rows in ``DatasetRegistry`` so
            the name-based eval/train flows and ``rllm dataset list`` work.

    Returns:
        Path to the built benchmark directory.
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[swebench_pro] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _purge_row_materialize_artifacts(out)
    backfilled = _patch_existing_dockerfiles(out)
    if backfilled:
        logger.info("[swebench_pro] backfilled ENTRYPOINT [] into %d existing Dockerfiles", backfilled)

    logger.info("[swebench_pro] loading HF dataset %s split=%s ...", HF_REPO_ID, hf_split)
    rows = _load_rows(hf_split)
    if task_ids is not None:
        keep = set(task_ids)
        rows = [r for r in rows if r.get("instance_id") in keep]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[swebench_pro] selected %d rows (task_ids=%s, limit=%s)", len(rows), task_ids and len(task_ids), limit)

    scripts_root = _shallow_clone_scripts_repo()
    try:
        written = 0
        skipped = 0
        for row in rows:
            instance_id = row.get("instance_id")
            if not instance_id:
                logger.warning("[swebench_pro] row missing instance_id, skipping")
                skipped += 1
                continue
            scripts_dir = _scripts_dir_for(scripts_root, instance_id)
            if scripts_dir is None:
                logger.warning("[swebench_pro] no run_scripts/ entry for %s, skipping", instance_id)
                skipped += 1
                continue
            if not (scripts_dir / "run_script.sh").is_file() or not (scripts_dir / "parser.py").is_file():
                logger.warning("[swebench_pro] %s missing run_script.sh or parser.py, skipping", instance_id)
                skipped += 1
                continue

            task_dst = out / instance_id
            if task_dst.exists():
                shutil.rmtree(task_dst)
            _materialize_task(task_dst, row, scripts_dir)
            written += 1

        description = (catalog_entry or {}).get("description") or "SWE-bench Pro (Public): 731 enterprise-grade SWE tasks across 41 repos, pre-built Docker images, F2P/P2P pytest-style grading."
        _write_dataset_toml(
            out,
            name=name,
            split=split,
            description=description,
            default_agent=default_agent,
        )
        logger.info("[swebench_pro] wrote %d task dirs to %s (skipped %d)", written, out, skipped)

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
                    instruction = (task_dst / "instruction.md").read_text(encoding="utf-8")
                    reg_rows.append(
                        {
                            "id": iid,
                            "instruction": instruction,
                            "task_path": str(task_dst),
                            "repo": row.get("repo", ""),
                            "repo_language": row.get("repo_language", ""),
                            "dockerhub_tag": row.get("dockerhub_tag", ""),
                        }
                    )
                DatasetRegistry.register_dataset(
                    name=name,
                    data=reg_rows,
                    split=split,
                    source=HF_REPO_ID,
                    description=description,
                    category=(catalog_entry or {}).get("category", "code"),
                )
            except Exception:
                logger.warning("[swebench_pro] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

        return out
    finally:
        shutil.rmtree(scripts_root, ignore_errors=True)


def main() -> None:
    """CLI: ``python -m rllm.data.swebench_pro_builder --out-dir <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize SWE-bench Pro into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="swebench_pro")
    parser.add_argument("--split", default="test")
    parser.add_argument("--hf-split", default="test")
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
        hf_split=args.hf_split,
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()
