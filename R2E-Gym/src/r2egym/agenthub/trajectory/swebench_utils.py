from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
from swebench.harness.grading import get_eval_tests_report, get_resolution_status

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)


def get_logs_eval(test_spec: TestSpec, content: str) -> tuple[dict[str, str], bool]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        log_fp (str): path to log file
    Returns:
        bool: whether the patch applied successfully
        dict: status map

    modified from swebench/harness/grading.py
    """
    repo = test_spec.repo
    version = test_spec.version
    log_parser = MAP_REPO_TO_PARSER[repo]
    test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    # with open(log_fp) as f:
    # # TODO fix constant here
    bad_codes = list(
        filter(
            lambda x: x in content,
            [
                APPLY_PATCH_FAIL,
                RESET_FAILED,
                TESTS_ERROR,
                TESTS_TIMEOUT,
            ],
        )
    )
    if bad_codes:
        return {}, False

    # elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
    #     # Test patch did not apply (should not happen at all)
    #     self.logger.error("Test patch did not apply")
    #     return {}, False

    # Get status map of evaluation results
    content = content.split(test_cmd)[-1]
    return log_parser(content, test_spec), True


def swebench_report(ds, test_output):
    test_spec = make_test_spec(ds)
    eval_status_map, found = get_logs_eval(test_spec, test_output)
    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    report = get_eval_tests_report(
        eval_status_map, eval_ref, eval_type=get_eval_type(test_spec)
    )
    return report


def swebench_parse(ds, test_output):
    test_spec = make_test_spec(ds)
    eval_status_map, found = get_logs_eval(test_spec, test_output)
    return eval_status_map
