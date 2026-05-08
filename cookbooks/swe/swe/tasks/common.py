"""Shared dataset-evaluation utilities."""

import ast
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path



KEY_INSTANCE_ID = "instance_id"
FAIL_TO_PASS = "FAIL_TO_PASS"
FAIL_TO_FAIL = "FAIL_TO_FAIL"
PASS_TO_PASS = "PASS_TO_PASS"
PASS_TO_FAIL = "PASS_TO_FAIL"



class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


class EvalType(Enum):
    PASS_AND_FAIL = "pass_and_fail"
    FAIL_ONLY = "fail_only"


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"



def make_log(verbose: bool):
    """Create a logging function gated on verbose flag."""
    if verbose:
        return print
    return lambda *_args, **_kwargs: None


def short_id(instance_id: str) -> str:
    """Derive a short display ID from a full instance_id."""
    if "." in instance_id:
        return instance_id.split(".")[-1][:8]
    if "-" in instance_id:
        return instance_id.split("-")[0]
    return instance_id[:30]


def zero_result(**overrides) -> dict:
    """Return a zero-reward grading result dict with optional overrides."""
    result = {"reward": 0.0, "f2p_passed": 0, "f2p_total": 0, "p2p_passed": 0, "p2p_total": 0}
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# Per-dataset fatal / invalid-patch logging.
#
# Each dataset writes failures to a pair of files at the repo root:
#   <dataset>_fatal_mistakes.txt   — infra/setup/grading errors
#   <dataset>_invalid_patches.txt  — patches that failed to apply
# The format is identical across datasets so a single analyzer script
# (swe/scripts/analyze_fatal_log.py) can consume any of them.
# ---------------------------------------------------------------------------


class PatchApplyError(RuntimeError):
    """Raised when a submitted patch is malformed or cannot be applied cleanly."""


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PATCH_PREVIEW_LINES = 80


def fatal_log_path(dataset: str, *, kind: str = "fatal", base_dir: Path | None = None) -> Path:
    """Return the per-dataset log path. ``kind`` is ``fatal`` or ``invalid_patch``."""
    suffix = "invalid_patches" if kind == "invalid_patch" else "fatal_mistakes"
    return (base_dir or _REPO_ROOT) / f"{dataset}_{suffix}.txt"


def append_fatal_log(
    dataset: str,
    stage: str,
    task: dict,
    short_id: str,
    message: str,
    patch: str = "",
    *,
    kind: str = "fatal",
    base_dir: Path | None = None,
) -> Path | None:
    """Append a failure entry to the dataset's fatal / invalid-patch log.

    Returns the path written to, or None on I/O failure. Failures writing to
    the log are swallowed so graders never crash on a disk issue.
    """
    log_path = fatal_log_path(dataset, kind=kind, base_dir=base_dir)
    lines = [
        "=" * 100,
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
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
        preview = "\n".join(patch.splitlines()[:_PATCH_PREVIEW_LINES])
        lines.extend(["patch_preview:", preview, ""])

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return log_path
    except OSError:
        return None


def parse_test_list(value) -> list:
    """Parse test list from list, JSON string, or Python literal string.

    DatasetRegistry may serialize lists as JSON strings,
    while CSV files use Python-style string lists.
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []

    value = value.strip()
    if not value or value == "[]":
        return []

    try:
        result = json.loads(value)
        return list(result) if isinstance(result, (list, set)) else []
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        result = ast.literal_eval(value)
        return list(result) if isinstance(result, (list, set, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def normalize_task_test_lists(task: dict, keys: tuple[str, ...] = ("FAIL_TO_PASS", "PASS_TO_PASS")) -> dict:
    """Ensure test list fields in a task dict are actual lists, not JSON strings.

    Returns a shallow copy of task with normalized fields.
    """
    task = dict(task)
    for key in keys:
        val = task.get(key, [])
        task[key] = parse_test_list(val) if val is not None else []
    return task


def has_new_file_in_patch(diff_text: str) -> bool:
    """Returns True if at least one file was newly added in the diff patch."""
    chunks = re.split(r'^diff --git ', diff_text, flags=re.MULTILINE)
    for chunk in chunks:
        if not chunk.strip():
            continue
        chunk = "diff --git " + chunk
        is_new_file = re.search(r'^new file mode \d+', chunk, flags=re.MULTILINE)
        has_dev_null = re.search(r'^--- /dev/null$', chunk, flags=re.MULTILINE)
        has_added_file = re.search(r'^\+\+\+ b/.*', chunk, flags=re.MULTILINE)
        if (is_new_file or has_dev_null) and has_added_file:
            return True
    return False



def test_passed(case: str, sm: dict[str, str]) -> bool:
    """Check if a test case passed."""
    return case in sm and sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]


def test_failed(case: str, sm: dict[str, str]) -> bool:
    """Check if a test case failed."""
    return case not in sm or sm[case] in [TestStatus.FAILED.value, TestStatus.ERROR.value, TestStatus.SKIPPED.value]


def get_resolution_status(report: dict) -> str:
    """Determine resolved status of an evaluation instance."""
    def compute_fail_to_pass(report: dict) -> float:
        total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
        if total == 0:
            return 1
        return len(report[FAIL_TO_PASS]["success"]) / total

    def compute_pass_to_pass(report: dict) -> float:
        total = len(report[PASS_TO_PASS]["success"]) + len(report[PASS_TO_PASS]["failure"])
        if total == 0:
            return 1
        return len(report[PASS_TO_PASS]["success"]) / total

    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    elif f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    else:
        return ResolvedStatus.NO.value


def get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, str],
    calculate_to_fail: bool = False,
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> dict[str, dict[str, list[str]]]:
    """Create a report based on failure/pass change from gold results to eval results."""

    def check_pass_and_fail(test_case, eval_status_map, success, failed):
        if test_passed(test_case, eval_status_map):
            success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            failed.append(test_case)

    def check_fail_only(test_case, eval_status_map, success, failed):
        if (
            test_case in eval_status_map
            and eval_status_map[test_case] == TestStatus.FAILED.value
        ):
            failed.append(test_case)
        else:
            success.append(test_case)

    check_test_case = (
        check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )

    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        check_test_case(test_case, eval_status_map, f2p_success, f2p_failure)

    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        check_test_case(test_case, eval_status_map, p2p_success, p2p_failure)

    results = {
        FAIL_TO_PASS: {"success": f2p_success, "failure": f2p_failure},
        PASS_TO_PASS: {"success": p2p_success, "failure": p2p_failure},
    }

    f2f_success, f2f_failure = [], []
    p2f_success, p2f_failure = [], []
    if calculate_to_fail:
        for test_case in gold_results.get(FAIL_TO_FAIL, []):
            check_test_case(test_case, eval_status_map, f2f_success, f2f_failure)
        for test_case in gold_results.get(PASS_TO_FAIL, []):
            check_test_case(test_case, eval_status_map, p2f_success, p2f_failure)

    results.update({
        FAIL_TO_FAIL: {"success": f2f_success, "failure": f2f_failure},
        PASS_TO_FAIL: {"success": p2f_success, "failure": p2f_failure},
    })
    return results
