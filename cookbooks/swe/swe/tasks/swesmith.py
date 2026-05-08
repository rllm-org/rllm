#!/usr/bin/env python3
"""SWE-smith grading using SWE-smith harness."""

import base64
import os
import re
import shlex
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS
from swebench.harness.grading import get_resolution_status, ResolvedStatus

from swe.tasks.common import (
    PatchApplyError,
    append_fatal_log,
    fatal_log_path,
    make_log,
    normalize_task_test_lists,
    short_id as _short_id,
    zero_result,
)

_DATASET = "swesmith"
_BASE_DIR = Path(__file__).parent.parent.parent
_SWE_SMITH_PATH = Path(os.environ.get("SWE_SMITH_PATH", _BASE_DIR / "external" / "SWE-smith"))

# Backwards-compat alias — some tests/helpers may still import this name.
ModelPatchApplyError = PatchApplyError

WORKDIR = "/testbed"
EVAL_PATH = "/tmp/eval.sh"

# Sandbox command timeouts (seconds).
GIT_FETCH_TIMEOUT = 180
GIT_CHECKOUT_TIMEOUT = 180
GIT_STAGE_TIMEOUT = 120
GIT_REINIT_TIMEOUT = 60
GIT_APPLY_TIMEOUT = 60
FILE_OP_TIMEOUT = 30
APT_INSTALL_TIMEOUT = 420
APT_LOCK_WAIT_TIMEOUT = 600
EVAL_SCRIPT_TIMEOUT = 600
COMMAND_CHECK_TIMEOUT = 30

# Transient failures we retry: Modal HTTP hiccups + apt lock / mirror sync issues.
# See swe/scripts/analyze_swesmith_fatal_log.py for the taxonomy.
_TRANSIENT_RE = re.compile(
    r"Connection timeout to host|Response payload is not completed|ContentLengthError|"
    r"Cannot connect to host|Command timed out after|ServerDisconnectedError|Server disconnected|ClientConnectorError|"
    r"Could not get lock|Unable to acquire the dpkg frontend lock|Failed to fetch|"
    r"Hash Sum mismatch|Mirror sync in progress"
)

# Interpreter shutdown race inside Modal's ThreadPoolExecutor: the driver
# process is exiting, atexit has already shut the pool down, and in-flight
# env.execute() calls fail to submit. This is not a sandbox/model error.
_SHUTDOWN_RE = re.compile(r"cannot schedule new futures after interpreter shutdown")


class InterpreterShutdown(RuntimeError):
    """Raised when env.execute fails because the interpreter is tearing down."""


@lru_cache(maxsize=1)
def _load_swesmith_components():
    """Import SWE-smith components lazily to avoid import-time side effects."""
    try:
        from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
        from swesmith.harness.grading import get_eval_tests_report
        from swesmith.profiles import registry

        return {
            "get_eval_tests_report": get_eval_tests_report,
            "registry": registry,
            "test_output_start": TEST_OUTPUT_START,
            "test_output_end": TEST_OUTPUT_END,
        }
    except ImportError:
        pass

    if not _SWE_SMITH_PATH.exists():
        raise ImportError(
            "SWE-smith grading requires the `swesmith` package or a local checkout. "
            "Install this cookbook with dependencies, or set SWE_SMITH_PATH."
        )

    if str(_SWE_SMITH_PATH) not in sys.path:
        sys.path.insert(0, str(_SWE_SMITH_PATH))

    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    from swesmith.harness.grading import get_eval_tests_report
    from swesmith.profiles import registry

    return {
        "get_eval_tests_report": get_eval_tests_report,
        "registry": registry,
        "test_output_start": TEST_OUTPUT_START,
        "test_output_end": TEST_OUTPUT_END,
    }


def _exec(env, cmd: str, *, cwd: str, timeout: int, check: bool = True, retries: int = 4) -> dict[str, Any]:
    """Run a sandbox command; retry on transient Modal/apt failures."""
    for attempt in range(retries):
        can_retry = attempt + 1 < retries
        try:
            res = env.execute({"command": cmd}, cwd=cwd, timeout=timeout)
        except Exception as exc:
            signal = f"{type(exc).__name__}: {exc}"
            if _SHUTDOWN_RE.search(signal):
                raise InterpreterShutdown(signal) from exc
            if can_retry and _TRANSIENT_RE.search(signal):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Command failed: {cmd}\ncwd={cwd}\ntimeout={timeout}s\n{signal}"
            ) from exc

        out = res.get("output", "") or ""
        if not check or res.get("returncode", 1) == 0:
            return res
        if can_retry and _TRANSIENT_RE.search(out):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"Command failed: {cmd}\n{out[:500]}")

    raise RuntimeError(f"Command failed: {cmd} (retry loop exhausted)")


def _write_b64(env, *, path: str, content: str, cwd: str) -> None:
    qpath = shlex.quote(path)
    _exec(env, f"rm -f {qpath}", cwd=cwd, timeout=FILE_OP_TIMEOUT, check=False)
    for i in range(0, len(content), 20000):
        chunk = content[i : i + 20000]
        enc = base64.b64encode(chunk.encode()).decode()
        _exec(
            env,
            f"printf '%s' {shlex.quote(enc)} | base64 -d >> {qpath}",
            cwd=cwd,
            timeout=FILE_OP_TIMEOUT,
            check=True,
        )


def _apply_patch(env, patch: str, working_dir: str, short_id: str) -> None:
    """Write patch to sandbox and apply it. Raises PatchApplyError on failure."""
    _write_b64(env, path="/tmp/agent_patch.diff", content=patch, cwd=working_dir)
    q = "/tmp/agent_patch.diff"

    res = _exec(env, f"git apply -v {q}", cwd=working_dir, timeout=GIT_APPLY_TIMEOUT, check=False)
    if res.get("returncode", 0) == 0:
        return

    res = _exec(env, f"git apply --reject {q}", cwd=working_dir, timeout=GIT_APPLY_TIMEOUT, check=False)
    if res.get("returncode", 0) != 0:
        raise PatchApplyError(
            f"[{short_id}] patch apply failed:\n{(res.get('output') or '')[:700]}"
        )

    rej = _exec(env, "find . -type f -name '*.rej' -print -quit", cwd=working_dir, timeout=FILE_OP_TIMEOUT, check=False)
    if (rej.get("output") or "").strip():
        raise PatchApplyError(f"[{short_id}] patch partially applied (.rej present)")


def _fetch_and_checkout(env, wd: str, iid: str) -> None:
    """Fetch + checkout as a unit, retrying the pair on pathspec errors from silent fetch failures."""
    quoted = shlex.quote(iid)
    for attempt in range(3):
        _exec(env, "git fetch --all --prune", cwd=wd, timeout=GIT_FETCH_TIMEOUT, check=False)
        try:
            _exec(env, f"git checkout {quoted}", cwd=wd, timeout=GIT_CHECKOUT_TIMEOUT)
            return
        except RuntimeError as exc:
            if attempt == 2 or "did not match any file" not in str(exc):
                raise
            time.sleep(2 ** attempt)


def _ensure_python3(env, wd: str) -> None:
    """Install python3 via apt if neither python3 nor python is already on PATH."""
    check = _exec(
        env,
        "command -v python3 >/dev/null || command -v python >/dev/null",
        cwd=wd, timeout=COMMAND_CHECK_TIMEOUT, check=False, retries=2,
    )
    if check.get("returncode", 1) == 0:
        return

    # Wait for unattended-upgrades to release the apt lock (433/857 errors
    # in the current log are "Could not get lock"), then install.
    install = (
        f"end=$(($(date +%s)+{APT_LOCK_WAIT_TIMEOUT})); "
        "while [ $(date +%s) -lt $end ]; do "
        "  pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null "
        "    || pgrep -x unattended-upgrade >/dev/null || break; "
        "  sleep 3; "
        "done; "
        "command -v apt-get >/dev/null || exit 127; "
        "DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 install -y -qq python3"
    )
    _exec(env, install, cwd=wd, timeout=APT_INSTALL_TIMEOUT + APT_LOCK_WAIT_TIMEOUT)


def setup_swesmith_agent_env(env, task: dict, short_id: str, log_fn=print) -> None:
    """Prepare SWE-smith agent sandbox: checkout instance branch, reinitialize git."""
    wd = task["working_dir"]
    iid = task["instance_id"]

    try:
        # HEAD already has F2P tests removed; P2P tests stay visible
        # (consistent with SWE-bench Pro/Multilingual)
        _fetch_and_checkout(env, wd, iid)

        # Reinitialize git so agent cannot recover tests from history
        _exec(env, "rm -rf .git", cwd=wd, timeout=GIT_REINIT_TIMEOUT)
        _exec(env, "git init", cwd=wd, timeout=GIT_REINIT_TIMEOUT)
        _exec(env, "git config user.email 'swesmith@local'", cwd=wd, timeout=FILE_OP_TIMEOUT)
        _exec(env, "git config user.name 'swesmith'", cwd=wd, timeout=FILE_OP_TIMEOUT)
        _exec(env, "git config commit.gpgsign false", cwd=wd, timeout=FILE_OP_TIMEOUT, check=False)
        _exec(env, "git add -A", cwd=wd, timeout=GIT_STAGE_TIMEOUT)
        _exec(env, "git commit --allow-empty -m 'agent-start'", cwd=wd, timeout=GIT_STAGE_TIMEOUT)

        _ensure_python3(env, wd)

        log_fn(f"[{short_id}] SWE-smith agent sandbox ready (git reinitialized)")
    except InterpreterShutdown:
        log_fn(f"[{short_id}] setup aborted: interpreter shutdown")
        raise
    except Exception:
        append_fatal_log(_DATASET, "setup_swesmith_env", task, short_id, traceback.format_exc())
        log_fn(f"[{short_id}] FATAL setup error: details appended to {fatal_log_path(_DATASET)}")
        raise


def _build_eval_script(profile, task: dict) -> str:
    """Build evaluation script using profile's test command."""
    components = _load_swesmith_components()
    test_cmd, _ = profile.get_test_cmd(task)
    return "\n".join([
        "#!/bin/bash",
        "set -uxo pipefail",
        f"cd {WORKDIR}",
        f": '{components['test_output_start']}'",
        test_cmd,
        f": '{components['test_output_end']}'",
    ]) + "\n"


def _parse_test_output(output: str) -> str:
    """Extract test output between markers."""
    components = _load_swesmith_components()
    start_marker = f"+ : '{components['test_output_start']}'"
    end_marker = f"+ : '{components['test_output_end']}'"

    start_idx = output.find(start_marker)
    end_idx = output.find(end_marker)

    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return output

    return output[start_idx + len(start_marker):end_idx]


def _restore_tests_for_grading(env, patch: str, working_dir: str, short_id: str, task: dict, _log) -> None:
    """Reset to HEAD~1 (bug patch with all tests) and apply model patch."""
    iid = task["instance_id"]

    _fetch_and_checkout(env, working_dir, iid)
    _exec(env, "git reset --hard HEAD~1", cwd=working_dir, timeout=GIT_REINIT_TIMEOUT)

    if patch and patch.strip():
        _apply_patch(env, patch, working_dir, short_id)

    _log(f"[{short_id}] Grading env ready")


def grade_swesmith_in_env(
    task: dict,
    env,
    patch: str,
    verbose: bool = False,
) -> dict:
    """Grade a SWE-smith patch in the Modal sandbox.

    Returns dict with reward (1.0 if resolved, 0.0 otherwise) and test details.
    """
    _log = make_log(verbose)
    fatal_logged = False

    task = normalize_task_test_lists(task, keys=("FAIL_TO_PASS", "PASS_TO_PASS", "fail_to_pass", "pass_to_pass"))
    sid = _short_id(task.get("instance_id", "unknown"))
    working_dir = task["working_dir"]

    try:
        _log(f"[{sid}] Starting evaluation...")

        components = _load_swesmith_components()
        profile = components["registry"].get_from_inst(task)

        try:
            _restore_tests_for_grading(env, patch, working_dir, sid, task, _log)
        except (PatchApplyError, InterpreterShutdown):
            raise
        except Exception:
            append_fatal_log(_DATASET, "_restore_tests_for_grading", task, sid, traceback.format_exc(), patch)
            fatal_logged = True
            raise

        eval_script = _build_eval_script(profile, task)
        env.execute({"command": f"cat > {EVAL_PATH} << 'EVAL_SCRIPT_EOF'\n{eval_script}\nEVAL_SCRIPT_EOF"}, cwd=working_dir, timeout=FILE_OP_TIMEOUT)
        env.execute({"command": f"chmod +x {EVAL_PATH}"}, cwd=working_dir, timeout=FILE_OP_TIMEOUT)

        result = env.execute({"command": f"cd {working_dir} && /bin/bash {EVAL_PATH}"}, cwd=working_dir, timeout=EVAL_SCRIPT_TIMEOUT)
        test_output = result.get("output", "")

        parsed_output = _parse_test_output(test_output)
        test_status = profile.log_parser(parsed_output)
        report = components["get_eval_tests_report"](test_status, task)
        resolved = get_resolution_status(report) == ResolvedStatus.FULL.value

        f2p = report[FAIL_TO_PASS]
        p2p = report[PASS_TO_PASS]
        f2p_pass, f2p_fail = len(f2p["success"]), len(f2p["failure"])
        p2p_pass, p2p_fail = len(p2p["success"]), len(p2p["failure"])
        _log(f"[{sid}] F2P {f2p_pass}/{f2p_pass + f2p_fail}, P2P {p2p_pass}/{p2p_pass + p2p_fail}")

        if resolved:
            _log(f"[{sid}] RESOLVED")
        else:
            failed = f2p["failure"] + p2p["failure"]
            for t in failed[:3]:
                _log(f"[{sid}]   - {t} (FAILED)")
            if len(failed) > 3:
                _log(f"[{sid}]   ... and {len(failed) - 3} more")

        return {
            "reward": 1.0 if resolved else 0.0,
            "f2p_passed": f2p_pass,
            "f2p_total": f2p_pass + f2p_fail,
            "p2p_passed": p2p_pass,
            "p2p_total": p2p_pass + p2p_fail,
        }

    except PatchApplyError as exc:
        invalid_path = fatal_log_path(_DATASET, kind="invalid_patch")
        append_fatal_log(_DATASET, "invalid_model_patch", task, sid, str(exc), patch, kind="invalid_patch")
        _log(f"[{sid}] Invalid patch; scoring 0.0 (logged to {invalid_path.name})")
        return zero_result(
            f2p_total=len(task.get(FAIL_TO_PASS, [])),
            p2p_total=len(task.get(PASS_TO_PASS, [])),
            invalid_patch=True,
            invalid_patch_reason=str(exc),
        )
    except InterpreterShutdown:
        _log(f"[{sid}] grading aborted: interpreter shutdown")
        raise
    except Exception:
        if not fatal_logged:
            append_fatal_log(_DATASET, "grade_swesmith", task, sid, traceback.format_exc(), patch)
        _log(f"[{sid}] FATAL: details appended to {fatal_log_path(_DATASET)}")
        raise


def grade_swesmith(
    task: dict,
    patch: str,
    env_factory: Callable[[dict], Any],
    verbose: bool = False,
) -> dict:
    """Create a fresh grading sandbox and evaluate a SWE-smith patch."""
    env = env_factory(task)
    try:
        return grade_swesmith_in_env(task, env, patch, verbose)
    finally:
        try:
            env.close()
        except Exception:
            pass
