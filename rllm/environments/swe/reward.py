"""Reward calculation functions for SWE environments.

Extracts reward logic from R2E-Gym's ExecutionEnvironment into standalone
functions that accept a SandboxSession and dataset dict.
"""

import json
import logging
import re


logger = logging.getLogger(__name__)


def parse_log_pytest(log: str | None) -> dict[str, str]:
    """
    Parser for test logs generated with pytest framework.

    Args:
        log: log content
    Returns:
        dict: test case to test status mapping
    """
    if log is None:
        return {}
    test_status_map = {}
    if "short test summary info" not in log:
        return test_status_map
    log = log.split("short test summary info")[1]
    log = log.strip()
    log = log.split("\n")
    for line in log:
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            try:
                test_name = ".".join(line.split("::")[1:])
            except IndexError:
                test_name = line
            test_name = test_name.split(" - ")[0]
            test_status_map[test_name] = "ERROR"
    return test_status_map


def decolor_dict_keys(key_dict) -> dict:
    """Remove ANSI escape codes from dictionary keys."""
    decolor = lambda key: re.sub(r"\u001b\[\d+m", "", key)
    return {decolor(k): v for k, v in key_dict.items()}


def calculate_reward(
    session, ds: dict, repo_path: str, alt_path: str, timeout: int = 300
) -> float:
    """Dispatch reward calculation by dataset type.

    Args:
        session: ARL SandboxSession instance.
        ds: Dataset entry dict.
        repo_path: Path to the repo inside the sandbox (e.g. /testbed).
        alt_path: Alternative path inside the sandbox (e.g. / or /root).
        timeout: Timeout for test execution in seconds.

    Returns:
        Reward value (0.0 or 1.0).
    """
    image = ds.get("docker_image", ds.get("image_name", ""))
    swebench_verified = "swebench" in image

    if swebench_verified:
        return _calculate_reward_swebench(session, ds, timeout=timeout)
    else:
        return _calculate_reward_r2e(session, ds, repo_path, alt_path, timeout=timeout)


def _run_in_session(session, cmd: str, workdir: str, timeout: int) -> tuple[str, str]:
    """Execute a command in the sandbox session.

    Returns:
        (output, error_code_str) matching the previous runtime interface.
    """
    response = session.execute(
        steps=[
            {
                "name": "reward_cmd",
                "command": ["sh", "-c", f"timeout {timeout} {cmd}"],
                "workDir": workdir,
                "timeout": timeout + 10,
            }
        ]
    )
    result = response.results[0]
    output = result.output.stdout
    if result.output.stderr:
        output = (
            output + "\n" + result.output.stderr if output else result.output.stderr
        )
    exit_code = result.output.exit_code

    output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

    if exit_code == 124:
        return f"The command took too long to execute (>{timeout}s)", "-1"
    if exit_code != 0:
        return output, f"Error: Exit code {exit_code}"
    return output, str(exit_code)


def _run_tests(session, alt_path: str, repo_path: str, timeout: int) -> tuple[str, str]:
    """Run the test script in the sandbox."""
    return _run_in_session(session, f"bash {alt_path}/run_tests.sh", repo_path, timeout)


def _calculate_reward_swebench(session, ds: dict, timeout: int = 300) -> float:
    """SWE-Bench Verified reward via swebench harness."""
    from swebench.harness.constants import (
        FAIL_TO_PASS,
        KEY_INSTANCE_ID,
        PASS_TO_PASS,
        ResolvedStatus,
    )
    from swebench.harness.grading import get_eval_tests_report, get_resolution_status
    from swebench.harness.log_parsers import get_eval_type
    from swebench.harness.test_spec import make_test_spec

    test_spec = make_test_spec(ds)

    out, _ = _run_in_session(session, "/run_tests.sh", "/testbed", timeout)
    eval_status_map, found = _get_logs_eval(test_spec, out)
    if not found:
        return 0.0

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    report = get_eval_tests_report(
        eval_status_map, eval_ref, eval_type=get_eval_type(test_spec)
    )
    success = get_resolution_status(report) == ResolvedStatus.FULL.value
    return int(success)


def _calculate_reward_r2e(
    session, ds: dict, repo_path: str, alt_path: str, timeout: int = 300
) -> float:
    """R2E reward via test output comparison."""
    output, _ = _run_tests(session, alt_path, repo_path, timeout)
    parse = parse_log_pytest(output)
    parse = decolor_dict_keys(parse)

    try:
        expected_json = ds["expected_output_json"]
    except (KeyError, TypeError):
        # Fallback: read from container
        expected_json, _ = _run_in_session(
            session, f"cat {alt_path}/expected_test_output.json", repo_path, 30
        )

    try:
        expected: dict = json.loads(expected_json)
    except (json.JSONDecodeError, TypeError):
        logger.error("Failed to parse expected output JSON")
        return 0.0
    expected = decolor_dict_keys(expected)
    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

    if len(parse) != len(expected):
        return 0.0

    for k in parse.keys():
        if not k:
            continue
        if k not in expected:
            return 0.0
        if parse[k] != expected[k]:
            return 0.0
    return 1.0


def _get_logs_eval(test_spec, content: str) -> tuple[dict[str, str], bool]:
    """Parse swebench evaluation logs."""
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        MAP_REPO_VERSION_TO_SPECS,
        RESET_FAILED,
        TESTS_ERROR,
        TESTS_TIMEOUT,
    )
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    repo = test_spec.repo
    version = test_spec.version
    log_parser = MAP_REPO_TO_PARSER[repo]
    test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    bad_codes = [
        x
        for x in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]
        if x in content
    ]
    if bad_codes:
        logger.error(f"Bad code found in log: {bad_codes}")
        return {}, False

    content = content.split(test_cmd)[-1]
    return log_parser(content, test_spec), True
