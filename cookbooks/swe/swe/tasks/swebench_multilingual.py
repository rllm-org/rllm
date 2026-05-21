#!/usr/bin/env python3
"""SWE-bench Multilingual grading using SWE-bench harness."""

import os
import tempfile
import traceback

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

from swe.tasks.common import (
    PatchApplyError,
    append_fatal_log,
    apply_patch_file,
    fatal_log_path,
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

_DATASET = "swebench_multilingual"

WORKDIR = "/testbed"
EVAL_PATH = "/tmp/eval_script.sh"


def setup_swebench_multilingual_agent_env(env, task: dict, short_id: str, log_fn=print) -> None:
    """Prepare Multilingual agent sandbox at base_commit with git history hidden."""
    working_dir = task.get("working_dir", WORKDIR)
    try:
        reset_repo(env, cwd=working_dir, ref=task["base_commit"], clean=True, runner=run_sandbox_with_retries)
        reinitialize_git_repo(
            env,
            cwd=working_dir,
            user_email="swebench-multilingual@local",
            user_name="swebench-multilingual",
            runner=run_sandbox_with_retries,
        )
        log_fn(f"[{short_id}] SWE-bench Multilingual agent sandbox ready (git reinitialized)")
    except Exception:
        append_fatal_log(_DATASET, "setup_swebench_multilingual_env", task, short_id, traceback.format_exc())
        log_fn(f"[{short_id}] FATAL setup error: details appended to {fatal_log_path(_DATASET)}")
        raise


def grade_swebench_multilingual_in_env(
    task: dict,
    env,
    patch: str,
    verbose: bool = False,
) -> dict:
    """Grade a multilingual patch in a fresh grading sandbox."""
    _log = make_log(verbose)

    instance_id = task.get("instance_id", "unknown")
    sid = _short_id(instance_id)
    working_dir = task.get("working_dir", WORKDIR)
    fail_to_pass = parse_test_list(task.get("FAIL_TO_PASS", task.get("fail_to_pass", [])))
    pass_to_pass = parse_test_list(task.get("PASS_TO_PASS", task.get("pass_to_pass", [])))

    _log(f"[{sid}] Starting evaluation...")

    try:
        # Fix /tmp permissions — Modal sandboxes set /tmp to 777 (world-writable
        # without sticky bit), which causes Ruby's Dir.tmpdir to refuse it.
        run_sandbox_with_retries(env, "chmod 1777 /tmp", cwd=working_dir, timeout=10)

        if patch and patch.strip():
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

        test_spec = make_test_spec(task, namespace="swebench")
        eval_script = test_spec.eval_script
        if not eval_script:
            raise RuntimeError("make_test_spec returned empty eval_script")

        write_file_b64(env, path=EVAL_PATH, content=eval_script, cwd=working_dir)
        run_sandbox_with_retries(env, f"chmod +x {EVAL_PATH}", cwd=working_dir, timeout=10)

        result = run_sandbox(env, f"bash {EVAL_PATH}", cwd=working_dir, timeout=1800, check=False)
        test_output = result["output"] or ""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_output)
            log_path = f.name

        try:
            prediction = {"instance_id": instance_id, "model_patch": patch}
            report = get_eval_report(test_spec, prediction, log_path, include_tests_status=True)
        finally:
            os.unlink(log_path)
    except Exception:
        append_fatal_log(_DATASET, "grade_swebench_multilingual", task, sid, traceback.format_exc(), patch)
        _log(f"[{sid}] FATAL grading error: details appended to {fatal_log_path(_DATASET)}")
        raise

    instance_report = report.get(instance_id, {})
    resolved = instance_report.get("resolved", False)
    tests_status = instance_report.get("tests_status", {})

    f2p = tests_status.get(FAIL_TO_PASS, {})
    p2p = tests_status.get(PASS_TO_PASS, {})
    f2p_pass = len(f2p.get('success', []))
    f2p_fail = len(f2p.get('failure', []))
    p2p_pass = len(p2p.get('success', []))
    p2p_fail = len(p2p.get('failure', []))

    _log(f"[{sid}] F2P {f2p_pass}/{f2p_pass + f2p_fail}, P2P {p2p_pass}/{p2p_pass + p2p_fail}")

    if resolved:
        _log(f"[{sid}] RESOLVED")
    else:
        failed_tests = f2p.get('failure', []) + p2p.get('failure', [])
        for t in failed_tests[:3]:
            _log(f"[{sid}]   - {t} (FAILED)")
        if len(failed_tests) > 3:
            _log(f"[{sid}]   ... and {len(failed_tests) - 3} more")

    return {
        "reward": 1.0 if resolved else 0.0,
        "f2p_passed": f2p_pass,
        "f2p_total": f2p_pass + f2p_fail,
        "p2p_passed": p2p_pass,
        "p2p_total": p2p_pass + p2p_fail,
    }
