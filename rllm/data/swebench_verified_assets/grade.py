"""In-sandbox grader for SWE-bench Verified.

Runs inside the task sandbox (copied to ``/tests/grade.py``). Extracts the
test-output section that swebench's eval script brackets with the
``>>>>> Start Test Output`` / ``>>>>> End Test Output`` markers, parses it
with the vendored per-repo parser (:mod:`swebench_parsers`), and applies
swebench's own resolution rule:

    passed  := status in {PASSED, XFAIL}
    resolved := every FAIL_TO_PASS test passed AND every PASS_TO_PASS passed

Writes the reward to a JSON file in the ShellScriptEvaluator contract shape.

Usage: ``python3 grade.py <eval_output.txt> <instance.json> <reward.json>``
"""

import json
import sys

sys.path.insert(0, "/tests")
import swebench_parsers as P  # noqa: E402  (resolved at runtime from /tests)

START_MARKER = ">>>>> Start Test Output"
END_MARKER = ">>>>> End Test Output"
PASS_STATUSES = ("PASSED", "XFAIL")


def _passed(case, status_map):
    """swebench's test_passed: present and PASSED or XFAIL."""
    return case in status_map and status_map[case] in PASS_STATUSES


def _extract_test_section(log):
    """Return the text swebench's eval script wrapped in the output markers.

    Falls back to the whole log when the markers are absent (the eval script
    aborted before reaching them) so the parser can still salvage any status
    lines that were emitted.
    """
    if START_MARKER in log and END_MARKER in log:
        return log.split(START_MARKER, 1)[1].rsplit(END_MARKER, 1)[0]
    return log


def main():
    eval_log_path, instance_json_path, reward_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(instance_json_path, encoding="utf-8") as f:
        inst = json.load(f)
    repo = inst.get("repo", "")
    f2p = list(inst.get("FAIL_TO_PASS") or [])
    p2p = list(inst.get("PASS_TO_PASS") or [])

    try:
        with open(eval_log_path, encoding="utf-8", errors="replace") as f:
            log = f.read()
    except OSError:
        log = ""

    section = _extract_test_section(log)
    parser = P.MAP_REPO_TO_PARSER.get(repo, P.parse_log_pytest)
    status_map = parser(section, None)

    f2p_missing = [c for c in f2p if not _passed(c, status_map)]
    p2p_missing = [c for c in p2p if not _passed(c, status_map)]

    # A valid SWE-bench task always declares FAIL_TO_PASS; treat an empty
    # F2P as unresolvable rather than vacuously solved.
    resolved = bool(f2p) and not f2p_missing and not p2p_missing
    reward = 1.0 if resolved else 0.0

    report = {
        "reward": reward,
        "is_correct": resolved,
        "signals": {
            "f2p_total": len(f2p),
            "f2p_passed": len(f2p) - len(f2p_missing),
            "p2p_total": len(p2p),
            "p2p_passed": len(p2p) - len(p2p_missing),
        },
        "metadata": {
            "repo": repo,
            "parser": getattr(parser, "__name__", "?"),
            "tests_parsed": len(status_map),
            "f2p_missing": f2p_missing[:50],
            "p2p_missing": p2p_missing[:50],
        },
    }
    with open(reward_path, "w", encoding="utf-8") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
