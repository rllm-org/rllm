#!/usr/bin/env python3
"""SWE-rebench V2 grading.

Applies test_patch in the agent's sandbox, runs test commands from
install_config, and parses output with the dataset's own log parsers.
"""

import base64
import importlib.util
import os
import re
import traceback
from functools import lru_cache
from pathlib import Path

from swe.tasks.common import (
    PatchApplyError,
    append_fatal_log, fatal_log_path,
    make_log, parse_test_list, short_id as _short_id, zero_result,
    KEY_INSTANCE_ID, FAIL_TO_PASS, PASS_TO_PASS,
    ResolvedStatus, EvalType,
    get_resolution_status, get_eval_tests_report,
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


def _write_b64(env, path: str, content: str, cwd: str) -> None:
    """Write content to a file in the sandbox via base64 to avoid shell escaping issues."""
    encoded = base64.b64encode(content.encode()).decode()
    chunk_size = 20000
    for i in range(0, len(encoded), chunk_size):
        chunk = encoded[i:i + chunk_size]
        op = ">>" if i > 0 else ">"
        env.execute({"command": f"echo -n '{chunk}' {op} {path}.b64"}, cwd=cwd, timeout=30)
    env.execute({"command": f"base64 -d {path}.b64 > {path} && rm {path}.b64"}, cwd=cwd, timeout=30)


def _apply_patch(env, patch_path: str, cwd: str, sid: str, label: str, log) -> None:
    """Apply a patch, trying clean apply first then falling back to --3way.

    Raises PatchApplyError if both attempts fail so the caller can classify
    the instance as invalid-patch and log to the dataset's invalid-patch file.
    """
    result = env.execute(
        {"command": f"git apply -v {patch_path}"},
        cwd=cwd, timeout=120,
    )
    if result.get("returncode", 0) == 0:
        log(f"[{sid}] {label} applied cleanly: {result.get('output', '')[:200]}")
        return

    result = env.execute(
        {"command": f"git apply -v --3way --recount --ignore-space-change --whitespace=nowarn {patch_path}"},
        cwd=cwd, timeout=120,
    )
    if result.get("returncode", 0) == 0:
        log(f"[{sid}] {label} applied (--3way): {result.get('output', '')[:200]}")
        return

    raise PatchApplyError(
        f"[{sid}] {label} apply failed:\n{(result.get('output') or '')[:700]}"
    )


def grade_swe_rebench_v2(
    task: dict,
    env,
    patch: str,
    verbose: bool = False,
) -> dict:
    """Grade a SWE-rebench V2 patch in the agent's sandbox.

    Applies test_patch, runs test commands, parses output with the dataset's
    log parsers, and computes F2P/P2P metrics.
    """
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
        # Reset to base commit, clean untracked files, then reapply patches.
        # Must use base_commit (not HEAD) because the agent may have committed.
        # Must clean untracked files because the agent may have created new files.
        base_commit = task.get("base_commit", "HEAD")
        env.execute({"command": f"git reset --hard {base_commit}"}, cwd=working_dir, timeout=30)
        env.execute({"command": "git clean -fd"}, cwd=working_dir, timeout=30)
        _write_b64(env, "/tmp/agent_patch.diff", patch, working_dir)
        _apply_patch(env, "/tmp/agent_patch.diff", working_dir, sid, "agent patch", _log)

        if test_patch and test_patch.strip():
            _write_b64(env, "/tmp/test_patch.diff", test_patch, working_dir)
            _apply_patch(env, "/tmp/test_patch.diff", working_dir, sid, "test_patch", _log)

        # Run test commands
        script = "set -e\n" + "\n".join(test_cmds)
        command = f"/bin/bash << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF"
        result = env.execute({"command": command}, cwd=working_dir, timeout=600)
        test_output = result.get("output", "")

        # Parse test output
        name_to_parser = _load_name_to_parser()
        if parser_name not in name_to_parser:
            print(f"[{sid}] WARNING: Unknown log parser '{parser_name}', returning zero reward")
            return {"reward": 0.0, "f2p_passed": 0, "f2p_total": len(fail_to_pass),
                    "p2p_passed": 0, "p2p_total": len(pass_to_pass)}

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
    except Exception:
        append_fatal_log(_DATASET, "grade_swe_rebench_v2", task, sid, traceback.format_exc(), patch)
        print(f"[{sid}] FATAL: details appended to {fatal_log_path(_DATASET)}")
        return zero_result()
