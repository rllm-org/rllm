from enum import Enum
from pathlib import Path
from pydantic import BaseModel

from r2e.paths import REPOS_DIR
from r2egym.repo_analysis.execution_log_parser import parse_log_fn


class CommitExecutionType(str, Enum):
    FAIL_AT_SETUP = "FAIL_AT_SETUP"
    FAIL_AT_TEST_RUN = "FAIL_AT_TEST_RUN"
    NEW_COMMIT_WORSE = "NEW_COMMIT_WORSE"
    NEW_COMMIT_SAME = "NEW_COMMIT_NOT_BETTER"
    NEW_COMMIT_BETTER = "NEW_COMMIT_BETTER"


class ExecutionResult(BaseModel):
    repo_name: str
    new_commit_hash: str

    test_file_codes: list[str]
    test_file_names: list[str]

    setup_res_code: int
    setup_res_stdout: str
    setup_res_stderr: str

    new_commit_res_code: int = -1
    new_commit_res_stdout: str | None = None
    new_commit_res_stderr: str | None = None

    old_commit_res_code: int = -1
    old_commit_res_stdout: str | None = None
    old_commit_res_stderr: str | None = None

    @property
    def old_commit_res_stdout_truncated(self) -> str:
        if len(self.old_commit_res_stdout) < 1000:
            return self.old_commit_res_stdout
        return "[TRUNCATED]\n\n" + self.old_commit_res_stdout[-1000:]

    @property
    def new_commit_res_stdout_truncated(self) -> str:
        if len(self.new_commit_res_stdout) < 1000:
            return self.new_commit_res_stdout
        return "[TRUNCATED]\n\n" + self.new_commit_res_stdout[-1000:]

    @property
    def old_commit_log_parse(self) -> dict:
        return parse_log_fn(self.repo_name)(self.old_commit_res_stdout)

    @property
    def new_commit_log_parse(self) -> dict:
        return parse_log_fn(self.repo_name)(self.new_commit_res_stdout)

    @property
    def new_repo_dir(self) -> Path:
        return REPOS_DIR / f"{self.repo_name}_{self.new_commit_hash}"

    def is_good_exec(self) -> tuple[CommitExecutionType, list[str]]:
        if self.setup_res_code != 0:
            return CommitExecutionType.FAIL_AT_SETUP, []
        old_test_parse = self.old_commit_log_parse
        new_test_parse = self.new_commit_log_parse

        if len(old_test_parse) != len(new_test_parse):
            return CommitExecutionType.FAIL_AT_TEST_RUN, []

        if len(old_test_parse) == 0:
            return CommitExecutionType.FAIL_AT_TEST_RUN, []

        ## ensure they have same keys
        if sorted(old_test_parse.keys()) != sorted(new_test_parse.keys()):
            return CommitExecutionType.FAIL_AT_TEST_RUN, []

        improvement_keys: list[str] = []

        for key in old_test_parse:
            if old_test_parse[key] == "PASSED":
                if new_test_parse[key] != "PASSED":
                    return CommitExecutionType.NEW_COMMIT_WORSE, []
            else:
                if new_test_parse[key] == "PASSED":
                    # NOTE: for parameterized tests we use
                    # the function name without the parameter
                    improvement_keys.append(key.split("[")[0])

        if improvement_keys:
            return CommitExecutionType.NEW_COMMIT_BETTER, improvement_keys

        return CommitExecutionType.NEW_COMMIT_SAME, improvement_keys

    def find_improved_tests_formatted(self) -> str:
        _, improved_fn_list = self.is_good_exec()
        if not improved_fn_list:
            return "No improvements found"
        output = ""
        for fn in set(improved_fn_list):
            output += f"- {fn}\n"
        return output
