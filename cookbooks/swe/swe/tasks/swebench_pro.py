#!/usr/bin/env python3
"""SWE-bench Pro grading using eval_with_modal."""

import os
import tempfile
import traceback
from pathlib import Path

from swe.environment import ensure_bootstrapped
from swe.tasks.common import (
    append_fatal_log,
    fatal_log_path,
    make_log,
    parse_test_list,
    short_id as _short_id,
    zero_result,
)

_DATASET = "swebench_pro"

# ensure_bootstrapped() adds external/SWE-bench_Pro-os to sys.path.
_BASE_DIR = Path(__file__).parent.parent.parent
_PRO_DIR = str(_BASE_DIR / "external" / "SWE-bench_Pro-os")

ensure_bootstrapped()

import swe_bench_pro_eval
from swe_bench_pro_eval import eval_with_modal

# Monkeypatch relative-path helpers to use absolute paths.
# The originals do open(f"dockerfiles/.../{iid}/Dockerfile") which requires
# os.chdir(_SWE_BENCH_PRO_PATH).  os.chdir is process-wide and not thread-safe,
# so we replace them with absolute-path versions instead.

def _load_base_docker(iid):
    with open(os.path.join(_PRO_DIR, f"dockerfiles/base_dockerfile/{iid}/Dockerfile")) as fp:
        return fp.read()

def _instance_docker(iid):
    with open(os.path.join(_PRO_DIR, f"dockerfiles/instance_dockerfile/{iid}/Dockerfile")) as fp:
        return fp.read()

swe_bench_pro_eval.load_base_docker = _load_base_docker
swe_bench_pro_eval.instance_docker = _instance_docker


def grade_swebench_pro(
    task: dict,
    patch: str,
    dockerhub_username: str,
    scripts_dir: str,
    verbose: bool = False,
) -> dict:
    """Grade a SWE-bench Pro patch using eval_with_modal.

    Returns dict with reward (1.0 if resolved, 0.0 otherwise) and test details.
    """
    _log = make_log(verbose)

    instance_id = task["instance_id"]
    sid = _short_id(instance_id)
    _log(f"[{sid}] Starting evaluation...")

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            output = eval_with_modal(
                patch=patch,
                sample=task,
                output_dir=output_dir,
                dockerhub_username=dockerhub_username,
                scripts_dir=scripts_dir,
                prefix="eval",
            )
    except Exception:
        append_fatal_log(_DATASET, "eval_with_modal", task, sid, traceback.format_exc(), patch)
        _log(f"[{sid}] FATAL eval_with_modal error: details appended to {fatal_log_path(_DATASET)}")
        raise

    if output is None:
        append_fatal_log(_DATASET, "eval_with_modal", task, sid, "eval_with_modal returned None", patch)
        _log(f"[{sid}] ERROR: Evaluation returned None (logged to {fatal_log_path(_DATASET).name})")
        return zero_result()

    passed_tests = {x["name"] for x in output.get("tests", []) if x.get("status") == "PASSED"}
    failed_tests = {x["name"] for x in output.get("tests", []) if x.get("status") == "FAILED"}

    f2p = set(parse_test_list(task.get("fail_to_pass", task.get("FAIL_TO_PASS", "[]"))))
    p2p = set(parse_test_list(task.get("pass_to_pass", task.get("PASS_TO_PASS", "[]"))))
    expected_tests = f2p | p2p

    f2p_passed = len(f2p & passed_tests)
    p2p_passed = len(p2p & passed_tests)

    _log(f"[{sid}] F2P {f2p_passed}/{len(f2p)}, P2P {p2p_passed}/{len(p2p)}")

    if not expected_tests:
        _log(f"[{sid}] ERROR: No expected tests in task definition")
        return zero_result()

    resolved = expected_tests <= passed_tests

    if resolved:
        _log(f"[{sid}] RESOLVED")
    else:
        not_passing = expected_tests - passed_tests
        for t in list(not_passing)[:3]:
            status = "FAILED" if t in failed_tests else "NOT FOUND"
            _log(f"[{sid}]   - {t} ({status})")
        if len(not_passing) > 3:
            _log(f"[{sid}]   ... and {len(not_passing) - 3} more")

    return {
        "reward": 1.0 if resolved else 0.0,
        "f2p_passed": f2p_passed,
        "f2p_total": len(f2p),
        "p2p_passed": p2p_passed,
        "p2p_total": len(p2p),
    }
