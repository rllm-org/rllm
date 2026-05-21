#!/usr/bin/env python3
"""SWE-rebench V2 grading."""

import importlib.util
import os
import re
import traceback
from functools import lru_cache
from pathlib import Path

from swe.tasks.common import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_PASS,
    EvalType,
    PatchApplyError,
    ResolvedStatus,
    append_fatal_log,
    apply_patch_file,
    fatal_log_path,
    get_eval_tests_report,
    get_resolution_status,
    make_log,
    parse_test_list,
    reinitialize_git_repo,
    reset_repo,
    run_sandbox,
    run_sandbox_with_retries,
    write_file_b64,
    zero_result,
)
from swe.tasks.common import (
    short_id as _short_id,
)

_DATASET = "swe_rebench_v2"

_BASE_DIR = Path(__file__).parent.parent.parent
_SWE_REBENCH_V2_PATH = Path(os.environ.get("SWE_REBENCH_V2_PATH", _BASE_DIR / "external" / "SWE-rebench-V2"))

# Regexes for stripping timing suffixes from test names (from rebench eval.py)
_TIMING_PATTERNS = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$"),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b"),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$"),
]


def _normalize_test_name(name: str) -> str:
    for pat in _TIMING_PATTERNS:
        name = pat.sub("", name)
    return name.strip()


@lru_cache(maxsize=1)
def _load_name_to_parser() -> dict:
    """Load NAME_TO_PARSER from SWE-rebench-V2's log_parsers module."""
    import sys

    if not _SWE_REBENCH_V2_PATH.exists():
        raise FileNotFoundError(
            "SWE-rebench V2 grading requires a local SWE-rebench-V2 checkout. "
            "Set SWE_REBENCH_V2_PATH to that checkout before evaluating swe_rebench_v2 tasks."
        )

    # The log_parsers module uses `from lib.agent.swe_constants import TestStatus`,
    # so we temporarily add the repo root to sys.path.
    repo_root = str(_SWE_REBENCH_V2_PATH)
    added = repo_root not in sys.path
    if added:
        sys.path.insert(0, repo_root)
    try:
        parsers_spec = importlib.util.spec_from_file_location(
            "swe_rebench_v2_log_parsers",
            _SWE_REBENCH_V2_PATH / "lib" / "agent" / "log_parsers.py",
        )
        parsers_mod = importlib.util.module_from_spec(parsers_spec)
        parsers_spec.loader.exec_module(parsers_mod)
        return parsers_mod.NAME_TO_PARSER
    finally:
        if added:
            sys.path.remove(repo_root)


def setup_swe_rebench_v2_agent_env(env, task: dict, short_id: str, log_fn=print) -> None:
    """Prepare Rebench V2 agent sandbox at base_commit with git history hidden."""
    working_dir = task["working_dir"]
    try:
        reset_repo(env, cwd=working_dir, ref=task["base_commit"], clean=True, runner=run_sandbox_with_retries)
        reinitialize_git_repo(
            env,
            cwd=working_dir,
            user_email="swe-rebench-v2@local",
            user_name="swe-rebench-v2",
            runner=run_sandbox_with_retries,
        )
        log_fn(f"[{short_id}] SWE-rebench V2 agent sandbox ready (git reinitialized)")
    except Exception:
        append_fatal_log(_DATASET, "setup_swe_rebench_v2_env", task, short_id, traceback.format_exc())
        log_fn(f"[{short_id}] FATAL setup error: details appended to {fatal_log_path(_DATASET)}")
        raise


def grade_swe_rebench_v2_in_env(
    task: dict,
    env,
    patch: str,
    verbose: bool = False,
) -> dict:
    """Grade a SWE-rebench V2 patch in a fresh grading sandbox."""
    _log = make_log(verbose)

    instance_id = task.get("instance_id", "unknown")
    sid = _short_id(instance_id)
    working_dir = task.get("working_dir", "/testbed")

    install_config = task.get("install_config", {})
    test_patch = task.get("test_patch", "")
    fail_to_pass = parse_test_list(task.get("fail_to_pass", task.get("FAIL_TO_PASS", [])))
    pass_to_pass = parse_test_list(task.get("pass_to_pass", task.get("PASS_TO_PASS", [])))

    test_cmds = install_config.get("test_cmd", [])
    if isinstance(test_cmds, str):
        test_cmds = [test_cmds]
    parser_name = install_config.get("log_parser", "")

    _log(f"[{sid}] Starting SWE-rebench V2 evaluation (parser={parser_name})...")

    try:
        # Fresh grading sandboxes are built with HEAD == base_commit; this mirrors
        # external/SWE-rebench-V2/scripts/eval.py. Do not reuse this in agent sandboxes.
        reset_repo(env, cwd=working_dir, ref="HEAD", clean=True, runner=run_sandbox_with_retries)
        write_file_b64(env, path="/tmp/agent_patch.diff", content=patch, cwd=working_dir)
        try:
            apply_patch_file(
                env,
                patch_path="/tmp/agent_patch.diff",
                cwd=working_dir,
                short_id=sid,
                label="agent patch",
                log_fn=_log,
            )
        except PatchApplyError as exc:
            invalid_path = fatal_log_path(_DATASET, kind="invalid_patch")
            append_fatal_log(_DATASET, "invalid_model_patch", task, sid, str(exc), patch, kind="invalid_patch")
            _log(f"[{sid}] Invalid patch; scoring 0.0 (logged to {invalid_path.name})")
            return zero_result(
                f2p_total=len(fail_to_pass),
                p2p_total=len(pass_to_pass),
                invalid_patch=True,
                invalid_patch_reason=str(exc),
            )

        if test_patch and test_patch.strip():
            write_file_b64(env, path="/tmp/test_patch.diff", content=test_patch, cwd=working_dir)
            try:
                apply_patch_file(
                    env,
                    patch_path="/tmp/test_patch.diff",
                    cwd=working_dir,
                    short_id=sid,
                    label="test_patch",
                    log_fn=_log,
                )
            except PatchApplyError as exc:
                raise RuntimeError(f"[{sid}] hidden test_patch failed to apply") from exc

        # Run test commands
        script = "set -e\n" + "\n".join(test_cmds)
        command = f"/bin/bash << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF"
        result = run_sandbox(env, command, cwd=working_dir, timeout=600, check=False)
        test_output = result["output"] or ""

        # Parse test output
        name_to_parser = _load_name_to_parser()
        if parser_name not in name_to_parser:
            raise KeyError(f"Unknown SWE-rebench V2 log parser: {parser_name!r}")

        parser_fn = name_to_parser[parser_name]
        test_status_map = parser_fn(test_output)

        # Normalize test names (strip timing suffixes)
        normalized_status = {_normalize_test_name(k): v for k, v in test_status_map.items()}
        normalized_f2p = [_normalize_test_name(t) for t in fail_to_pass]
        normalized_p2p = [_normalize_test_name(t) for t in pass_to_pass]

        eval_ref = {
            KEY_INSTANCE_ID: instance_id,
            FAIL_TO_PASS: normalized_f2p,
            PASS_TO_PASS: normalized_p2p,
        }
        report = get_eval_tests_report(normalized_status, eval_ref, eval_type=EvalType.PASS_AND_FAIL)
        resolved = get_resolution_status(report) == ResolvedStatus.FULL.value

        f2p_success = len(report.get(FAIL_TO_PASS, {}).get("success", []))
        p2p_success = len(report.get(PASS_TO_PASS, {}).get("success", []))
        _log(f"[{sid}] Grading: F2P {f2p_success}/{len(normalized_f2p)} passed, "
             f"P2P {p2p_success}/{len(normalized_p2p)} passed")

        if resolved:
            _log(f"[{sid}] RESOLVED - All tests passed")

        return {
            "reward": 1.0 if resolved else 0.0,
            "f2p_passed": f2p_success,
            "f2p_total": len(normalized_f2p),
            "p2p_passed": p2p_success,
            "p2p_total": len(normalized_p2p),
        }

    except Exception:
        append_fatal_log(_DATASET, "grade_swe_rebench_v2", task, sid, traceback.format_exc(), patch)
        _log(f"[{sid}] FATAL: details appended to {fatal_log_path(_DATASET)}")
        raise
