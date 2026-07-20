"""Builder for the SWE-bench Verified sandbox benchmark.

SWE-bench Verified (``princeton-nlp/SWE-bench_Verified``) is the 500-instance
human-filtered subset of SWE-bench: real GitHub issues across 12 Python repos
(django, sympy, astropy, matplotlib, sphinx, scikit-learn, …). Each HF row
carries the problem statement, the gold ``patch``, the ``test_patch`` that adds
the checking tests, and the ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` node ids.

Unlike SWE-bench Pro, the per-repo test command is non-trivial (django uses
``runtests.py``, sympy ``bin/test``, others pytest with repo-specific options)
and the log formats differ per repo. Rather than reinvent that, this builder
uses the ``swebench`` package (the ``swe`` optional extra) at *build time* to
generate, per instance:

  * the exact prebuilt Docker image (``make_test_spec(..., namespace="swebench")``
    → ``swebench/sweb.eval.x86_64.<id>:latest``), and
  * the eval script (``TestSpec.eval_script``) that reinstalls the repo, resets
    the test files, applies the gold ``test_patch``, and runs the repo's own
    test command, bracketing the output with swebench's Start/End markers.

Grading stays fully in-sandbox (no ``swebench`` at eval time): the vendored
per-repo log parser (``swebench_verified_assets/swebench_parsers.py``) parses
the marked section and ``grade.py`` applies swebench's resolution rule (every
FAIL_TO_PASS and PASS_TO_PASS test must pass). This expands each row into
rLLM's sandbox (task-per-directory) shape so ``rllm eval`` runs it through the
standard ``SandboxedAgentFlow`` + ``ShellScriptEvaluator`` path.

On-disk output (``<out_dir>/``)::

    swebench_verified/
    ├── dataset.toml                       # type="sandbox"
    ├── <instance_id>/
    │   ├── task.toml                      # docker_image=swebench/sweb.eval..., workdir=/testbed
    │   ├── instruction.md                 # problem_statement
    │   ├── environment/Dockerfile         # FROM <image> + ENTRYPOINT []
    │   ├── tests/
    │   │   ├── test.sh                    # static verifier orchestration
    │   │   ├── eval.sh                    # per-instance swebench eval script
    │   │   ├── grade.py                   # static: parse markers + score
    │   │   ├── swebench_parsers.py        # static: vendored swebench log parsers
    │   │   └── instance.json             # repo, base_commit, F2P/P2P
    │   └── solution/
    │       ├── gold.patch                 # the reference patch
    │       └── solve.sh                   # apply gold.patch (oracle harness)
    └── ...

Invoked from ``rllm dataset pull swebench_verified`` via the ``builder`` field
in ``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_REPO_ID = "princeton-nlp/SWE-bench_Verified"
# SWE-bench's public prebuilt eval images live under docker.io/swebench/.
# Passing namespace to make_test_spec also applies the ``__`` -> ``_1776_``
# tag rewrite the remote images use.
IMAGE_NAMESPACE = "swebench"
WORKDIR = "/testbed"

_ASSETS_DIR = Path(__file__).parent / "swebench_verified_assets"
_STATIC_TEST_FILES = ("test.sh", "grade.py", "swebench_parsers.py")

# Per-instance resource defaults. The sweb.eval images bundle the full repo +
# its test deps; django/sympy/matplotlib suites are the heavy end. Docker
# ignores these; Modal/Daytona honor them.
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


def _lazy_make_test_spec():
    """Import ``swebench.make_test_spec`` with a clear install hint if missing."""
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as e:
        raise RuntimeError("Building swebench_verified needs the `swebench` package. Install the SWE extra: `pip install 'rllm[swe]'` (or `pip install swebench`).") from e
    return make_test_spec


def _decode_json_list(value: Any) -> list[str]:
    """Parse a field that's either a JSON-encoded list or already a list.

    SWE-bench stores ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` as JSON-string lists
    on HF (e.g. ``'["a", "b"]'``); ``datasets`` sometimes hands them back
    already parsed.
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
            logger.warning("[swebench_verified] could not parse list field: %r", text[:120])
            return []
    return [str(v) for v in (loaded or [])]


def _build_instruction(row: dict) -> str:
    """The agent brief is the raw GitHub issue (SWE-bench's problem_statement)."""
    return (row.get("problem_statement") or "").strip() + "\n"


def _build_dockerfile(image: str) -> str:
    """``FROM <image>`` + clear ENTRYPOINT so rLLM's ``sleep infinity`` start
    command is honored (the base images set ``CMD``/``ENTRYPOINT`` to bash,
    which would otherwise eat the start command and exit the container).
    """
    return f"FROM {image}\nENTRYPOINT []\nWORKDIR {WORKDIR}\n"


def _build_task_toml(
    *,
    instance_id: str,
    repo: str,
    version: str,
    base_commit: str,
    image: str,
) -> str:
    """Synthesize a Harbor-format ``task.toml``.

    The loader lifts ``[environment].docker_image`` / ``workdir`` / resources
    into ``task.metadata`` so non-docker backends pull the prebuilt image
    rather than rebuilding the Dockerfile.
    """
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "swebench_verified/{instance_id}"',
        f'description = "SWE-bench Verified: {repo}"',
        'keywords = ["swe-bench", "swe-bench-verified", "python"]',
        "",
        "[metadata]",
        f'instance_id = "{instance_id}"',
        f'repo = "{repo}"',
        f'version = "{version}"',
        f'base_commit = "{base_commit}"',
        f'docker_image = "{image}"',
        "",
        "[environment]",
        f'docker_image = "{image}"',
        f'workdir = "{WORKDIR}"',
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


def _build_solution_script(base_commit: str) -> str:
    """``solution/solve.sh`` applies the gold patch — used by the ``oracle`` harness.

    Hard-resets to ``base_commit`` first: the sweb.eval images don't guarantee
    ``/testbed``'s HEAD is at base_commit, and the gold patch is expressed
    against it.
    """
    return (
        "#!/bin/bash\n"
        "set -e\n"
        f"cd {WORKDIR}\n"
        f"git config --global --add safe.directory {WORKDIR} 2>/dev/null || true\n"
        f'git reset --hard "{base_commit}"\n'
        f'git checkout "{base_commit}"\n'
        "git apply -v /solution/gold.patch\n"
    )


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


def _copy_static_assets(tests_dst: Path) -> None:
    """Copy the static verifier files (test.sh, grade.py, swebench_parsers.py)."""
    for fname in _STATIC_TEST_FILES:
        src = _ASSETS_DIR / fname
        dst = tests_dst / fname
        shutil.copy2(src, dst)
    (tests_dst / "test.sh").chmod(0o755)


def _purge_row_materialize_artifacts(out: Path) -> None:
    """Remove artifacts from the old HF-row materialize path.

    ``swebench_verified`` previously materialized as an HF-row dataset (a
    ``swebench_transform`` + ``swebench_reward_fn`` that no longer resolves).
    That left ``data/``, ``*.parquet``, ``instruction.md.tpl`` and a
    transform-style ``dataset.toml`` under ``~/.rllm/datasets/swebench_verified``.
    ``BenchmarkLoader`` routes through the row path when it sees ``data/``, so
    wipe the conflicting paths up front to make re-pulls converge on the
    sandbox shape.
    """
    for stale in ("data", "images", "instruction.md.tpl", "test.parquet", "test_verl.parquet"):
        target = out / stale
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            try:
                target.unlink()
            except OSError:
                pass


def _materialize_task(task_dir: Path, row: dict, make_test_spec) -> dict:
    """Expand a single HF row into a Harbor-format task tree. Returns stats."""
    task_dir.mkdir(parents=True, exist_ok=True)

    instance_id = row["instance_id"]
    repo = row.get("repo", "")
    base_commit = row.get("base_commit", "")

    # swebench builds the correct image name (with the swebench namespace + the
    # ``_1776_`` tag rewrite) and the per-repo eval script.
    ts = make_test_spec(row, namespace=IMAGE_NAMESPACE)
    image = ts.instance_image_key
    version = str(getattr(ts, "version", row.get("version", "")))
    eval_script = ts.eval_script

    # task.toml
    (task_dir / "task.toml").write_text(
        _build_task_toml(
            instance_id=instance_id,
            repo=repo,
            version=version,
            base_commit=base_commit,
            image=image,
        ),
        encoding="utf-8",
    )

    # instruction.md
    (task_dir / "instruction.md").write_text(_build_instruction(row), encoding="utf-8")

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(image), encoding="utf-8")

    # tests/
    tests_dst = task_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    _copy_static_assets(tests_dst)
    (tests_dst / "eval.sh").write_text(eval_script, encoding="utf-8")
    (tests_dst / "eval.sh").chmod(0o755)

    f2p = _decode_json_list(row.get("FAIL_TO_PASS"))
    p2p = _decode_json_list(row.get("PASS_TO_PASS"))
    instance_data = {
        "instance_id": instance_id,
        "repo": repo,
        "version": version,
        "base_commit": base_commit,
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
    }
    (tests_dst / "instance.json").write_text(json.dumps(instance_data, indent=2), encoding="utf-8")

    # solution/ (gold patch + apply script for the oracle harness)
    sol_dst = task_dir / "solution"
    sol_dst.mkdir(parents=True, exist_ok=True)
    (sol_dst / "gold.patch").write_text(row.get("patch") or "", encoding="utf-8")
    (sol_dst / "solve.sh").write_text(_build_solution_script(base_commit), encoding="utf-8")
    (sol_dst / "solve.sh").chmod(0o755)

    return {"f2p": len(f2p), "p2p": len(p2p), "image": image}


def _load_rows(hf_split: str) -> list[dict]:
    """Load the HF dataset rows as a list of plain dicts."""
    from datasets import load_dataset

    ds = load_dataset(HF_REPO_ID, split=hf_split)
    return [dict(r) for r in ds]


def build_benchmark(
    *,
    name: str = "swebench_verified",
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
    """Materialize SWE-bench Verified into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: Split label written into dataset.toml and the registry.
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``
            and ``default_agent`` are read from it when present.
        task_ids: Build only these ``instance_id`` values. Default: all 500.
        limit: Keep only the first N rows (after the ``task_ids`` filter).
        default_agent: ``default_agent`` written into dataset.toml.
        hf_split: HF split to load (``test`` — the only split in Verified).
        clean: Remove ``out_dir`` before building.
        register: Also register ``task_path`` rows in ``DatasetRegistry`` so
            the name-based eval/train flows and ``rllm dataset list`` work.

    Returns:
        Path to the built benchmark directory.
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent

    make_test_spec = _lazy_make_test_spec()

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[swebench_verified] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _purge_row_materialize_artifacts(out)

    logger.info("[swebench_verified] loading HF dataset %s split=%s ...", HF_REPO_ID, hf_split)
    rows = _load_rows(hf_split)
    if task_ids is not None:
        keep = set(task_ids)
        rows = [r for r in rows if r.get("instance_id") in keep]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[swebench_verified] selected %d rows (task_ids=%s, limit=%s)", len(rows), task_ids and len(task_ids), limit)

    written = 0
    skipped = 0
    for row in rows:
        instance_id = row.get("instance_id")
        if not instance_id:
            logger.warning("[swebench_verified] row missing instance_id, skipping")
            skipped += 1
            continue
        try:
            task_dst = out / instance_id
            if task_dst.exists():
                shutil.rmtree(task_dst)
            _materialize_task(task_dst, row, make_test_spec)
            written += 1
        except Exception:
            logger.warning("[swebench_verified] failed to materialize %s, skipping", instance_id, exc_info=True)
            skipped += 1

    description = (catalog_entry or {}).get("description") or (
        "SWE-bench Verified: 500 human-validated real-world GitHub issues across 12 Python repos (Harbor format; pre-built swebench/sweb.eval images; in-sandbox F2P/P2P grading)."
    )
    _write_dataset_toml(
        out,
        name=name,
        split=split,
        description=description,
        default_agent=default_agent,
    )
    logger.info("[swebench_verified] wrote %d task dirs to %s (skipped %d)", written, out, skipped)

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
            logger.warning("[swebench_verified] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out


def main() -> None:
    """CLI: ``python -m rllm.data.swebench_verified_builder --out-dir <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize SWE-bench Verified into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="swebench_verified")
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
