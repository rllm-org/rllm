#!/usr/bin/env python3
"""SWE-bench Multilingual grading using SWE-bench harness."""

import base64
import os
import shlex
import tempfile

from tasks.common import make_log, short_id as _short_id, zero_result
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.grading import get_eval_report
from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS

WORKDIR = "/testbed"
EVAL_PATH = "/tmp/eval_script.sh"


def _write_file(env, path: str, content: str, working_dir: str) -> None:
    """Write content to file using base64 encoding for robustness."""
    qpath = shlex.quote(path)
    env.execute({"command": f"rm -f {qpath}"}, cwd=working_dir, timeout=10)
    for i in range(0, len(content), 20000):
        chunk = content[i:i + 20000]
        encoded = base64.b64encode(chunk.encode()).decode()
        env.execute(
            {"command": f"printf '%s' {shlex.quote(encoded)} | base64 -d >> {qpath}"},
            cwd=working_dir, timeout=30,
        )


def grade_swebench_multilingual(
    task: dict,
    env,
    patch: str,
    verbose: bool = False,
) -> dict:
    """Grade a multilingual patch using SWE-bench harness.

    The eval_script from make_test_spec handles test file reset,
    test_patch application, build, test execution, and cleanup.

    Returns dict with reward (1.0 if resolved, 0.0 otherwise) and test details.
    """
    _log = make_log(verbose)

    instance_id = task.get("instance_id", "unknown")
    sid = _short_id(instance_id)
    working_dir = task.get("working_dir", WORKDIR)

    _log(f"[{sid}] Starting evaluation...")

    if env is None:
        _log(f"[{sid}] ERROR: No active environment available for multilingual grading")
        return zero_result()

    # Fix /tmp permissions — Modal sandboxes set /tmp to 777 (world-writable
    # without sticky bit), which causes Ruby's Dir.tmpdir to refuse it.
    env.execute({"command": "chmod 1777 /tmp"}, cwd=working_dir, timeout=10)

    test_spec = make_test_spec(task, namespace="swebench")
    eval_script = test_spec.eval_script
    if not eval_script:
        _log(f"[{sid}] ERROR: Empty eval_script from make_test_spec")
        return zero_result()

    _write_file(env, EVAL_PATH, eval_script, working_dir)
    env.execute({"command": f"chmod +x {EVAL_PATH}"}, cwd=working_dir, timeout=10)

    result = env.execute({"command": f"bash {EVAL_PATH}"}, cwd=working_dir, timeout=1800)
    test_output = result.get("output", "")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_output)
        log_path = f.name

    try:
        prediction = {"instance_id": instance_id, "model_patch": patch}
        report = get_eval_report(test_spec, prediction, log_path, include_tests_status=True)
    finally:
        os.unlink(log_path)

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
