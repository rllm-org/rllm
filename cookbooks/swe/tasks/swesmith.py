#!/usr/bin/env python3
"""SWE-smith grading using SWE-smith harness."""

import base64
import shlex
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS
from swebench.harness.grading import get_resolution_status, ResolvedStatus

from tasks.common import make_log, normalize_task_test_lists, short_id as _short_id, zero_result

_BASE_DIR = Path(__file__).parent.parent
_FATAL_LOG_PATH = _BASE_DIR / "swesmith_fatal_mistakes.txt"
_INVALID_PATCH_LOG_PATH = _BASE_DIR / "swesmith_invalid_patches.txt"

WORKDIR = "/testbed"
EVAL_PATH = "/tmp/eval.sh"

# Sandbox command timeouts (seconds)
GIT_FETCH_TIMEOUT = 120
GIT_CHECKOUT_TIMEOUT = 120
GIT_STAGE_TIMEOUT = 120
GIT_REINIT_TIMEOUT = 60
GIT_APPLY_TIMEOUT = 60
FILE_OP_TIMEOUT = 30
APT_INSTALL_TIMEOUT = 120
EVAL_SCRIPT_TIMEOUT = 600
COMMAND_CHECK_TIMEOUT = 5


class ModelPatchApplyError(RuntimeError):
    """Raised when the submitted model patch is malformed or cannot be applied."""


@lru_cache(maxsize=1)
def _load_swesmith_components():
    """Import SWE-smith components lazily to avoid import-time side effects."""

    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    from swesmith.harness.grading import get_eval_tests_report
    from swesmith.profiles import registry

    return {
        "get_eval_tests_report": get_eval_tests_report,
        "registry": registry,
        "test_output_start": TEST_OUTPUT_START,
        "test_output_end": TEST_OUTPUT_END,
    }


def _exec(env, cmd: str, *, cwd: str, timeout: int, check: bool = True) -> dict[str, Any]:
    try:
        res = env.execute({"command": cmd}, cwd=cwd, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(
            f"Command failed: {cmd}\ncwd={cwd}\ntimeout={timeout}s\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if check and res.get("returncode", 1) != 0:
        out = (res.get("output", "") or "")[:500]
        raise RuntimeError(f"Command failed: {cmd}\n{out}")
    return res


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


def _append_fatal_log(
    stage: str,
    task: dict,
    short_id: str,
    message: str,
    patch: str = "",
    log_path: Path | None = None,
) -> None:
    """Append fatal SWE-smith grading/setup failures for post-run debugging."""
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [
        "=" * 100,
        f"timestamp_utc: {timestamp}",
        f"stage: {stage}",
        f"instance_id: {task.get('instance_id', 'unknown')}",
        f"short_id: {short_id}",
        f"repo: {task.get('repo', 'unknown')}",
        f"working_dir: {task.get('working_dir', 'unknown')}",
        f"docker_image: {task.get('docker_image', 'unknown')}",
        f"patch_chars: {len(patch or '')}",
        "error_trace:",
        message.rstrip(),
        "",
    ]
    if patch.strip():
        lines.extend(["patch_preview:", "\n".join(patch.splitlines()[:80]), ""])

    try:
        target_path = log_path or _FATAL_LOG_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def _apply_patch(env, patch: str, working_dir: str, short_id: str) -> None:
    """Write patch to sandbox and apply it. Raises ModelPatchApplyError on failure."""
    _write_b64(env, path="/tmp/agent_patch.diff", content=patch, cwd=working_dir)
    q = "/tmp/agent_patch.diff"

    res = _exec(env, f"git apply -v {q}", cwd=working_dir, timeout=GIT_APPLY_TIMEOUT, check=False)
    if res.get("returncode", 0) == 0:
        return

    res = _exec(env, f"git apply --reject {q}", cwd=working_dir, timeout=GIT_APPLY_TIMEOUT, check=False)
    if res.get("returncode", 0) != 0:
        raise ModelPatchApplyError(
            f"[{short_id}] patch apply failed:\n{(res.get('output') or '')[:700]}"
        )

    rej = _exec(env, "find . -type f -name '*.rej' -print -quit", cwd=working_dir, timeout=FILE_OP_TIMEOUT, check=False)
    if (rej.get("output") or "").strip():
        raise ModelPatchApplyError(f"[{short_id}] patch partially applied (.rej present)")


def setup_swesmith_agent_env(env, task: dict, short_id: str, log_fn=print) -> None:
    """Prepare SWE-smith agent sandbox: checkout instance branch, reinitialize git."""
    wd = task["working_dir"]
    iid = task["instance_id"]

    try:
        # HEAD already has F2P tests removed; P2P tests stay visible
        # (consistent with SWE-bench Pro/Multilingual)
        _exec(env, "git fetch --all --prune", cwd=wd, timeout=GIT_FETCH_TIMEOUT, check=False)
        _exec(env, f"git checkout {shlex.quote(iid)}", cwd=wd, timeout=GIT_CHECKOUT_TIMEOUT)

        # Reinitialize git so agent cannot recover tests from history
        _exec(env, "rm -rf .git", cwd=wd, timeout=GIT_REINIT_TIMEOUT)
        _exec(env, "git init", cwd=wd, timeout=GIT_REINIT_TIMEOUT)
        _exec(env, "git config user.email 'swesmith@local'", cwd=wd, timeout=FILE_OP_TIMEOUT)
        _exec(env, "git config user.name 'swesmith'", cwd=wd, timeout=FILE_OP_TIMEOUT)
        _exec(env, "git config commit.gpgsign false", cwd=wd, timeout=FILE_OP_TIMEOUT, check=False)
        _exec(env, "git add -A", cwd=wd, timeout=GIT_STAGE_TIMEOUT)
        _exec(env, "git commit --allow-empty -m 'agent-start'", cwd=wd, timeout=GIT_STAGE_TIMEOUT)

        # Ensure python3 is available for eval scripts
        py = _exec(env, "command -v python3 >/dev/null 2>&1", cwd=wd, timeout=COMMAND_CHECK_TIMEOUT, check=False)
        if py.get("returncode", 1) != 0:
            _exec(
                env,
                "if command -v apt-get >/dev/null 2>&1; then "
                "DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3; "
                "else exit 127; fi",
                cwd=wd,
                timeout=APT_INSTALL_TIMEOUT,
            )

        log_fn(f"[{short_id}] SWE-smith agent sandbox ready (git reinitialized)")
    except Exception:
        _append_fatal_log("setup_swesmith_env", task, short_id, traceback.format_exc())
        log_fn(f"[{short_id}] FATAL setup error: details appended to {_FATAL_LOG_PATH}")
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

    _exec(env, "git fetch --all --prune", cwd=working_dir, timeout=GIT_FETCH_TIMEOUT, check=False)
    _exec(env, f"git checkout {shlex.quote(iid)}", cwd=working_dir, timeout=GIT_CHECKOUT_TIMEOUT)
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
        except ModelPatchApplyError:
            raise
        except Exception:
            _append_fatal_log("_restore_tests_for_grading", task, sid, traceback.format_exc(), patch)
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

    except ModelPatchApplyError as exc:
        _append_fatal_log("invalid_model_patch", task, sid, str(exc), patch, log_path=_INVALID_PATCH_LOG_PATH)
        _log(f"[{sid}] Invalid patch; scoring 0.0 (logged to {_INVALID_PATCH_LOG_PATH.name})")
        return zero_result(
            f2p_total=len(task.get(FAIL_TO_PASS, [])),
            p2p_total=len(task.get(PASS_TO_PASS, [])),
            invalid_patch=True,
            invalid_patch_reason=str(exc),
        )
    except Exception:
        if not fatal_logged:
            _append_fatal_log("grade_swesmith", task, sid, traceback.format_exc(), patch)
        _log(f"[{sid}] FATAL: details appended to {_FATAL_LOG_PATH}")
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
