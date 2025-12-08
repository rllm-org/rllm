import json

import pandas as pd

from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.repo_analysis.execution_result_analysis import ExecutionResult

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("r2e-edits/r2e-dockers-v1")["train"].to_pandas()


improved_tests = (
    ds["execution_result_content"]
    .apply(lambda x: ExecutionResult(**json.loads(x)))
    .apply(lambda x: x.is_good_exec()[1])
    .apply(lambda x: len(x))
)
print(pd.Series(improved_tests).describe())

parsed_commits: list[ParsedCommit] = (
    ds["parsed_commit_content"].apply(lambda x: ParsedCommit(**json.loads(x))).tolist()
)

print(f"Loaded {len(parsed_commits)} parsed commits")
non_test_lines = [c.get_num_lines_edited(False) for c in parsed_commits]
print(pd.Series(non_test_lines).describe())

test_lines = [c.get_num_lines_edited(True, False) for c in parsed_commits]
print(pd.Series(test_lines).describe())

issues = (
    ds["problem_statement"]
    .apply(lambda x: x.split("[/ISSUE]")[0])
    .apply(lambda x: x.split("[ISSUE]")[1] if "[ISSUE]" in x else x)
    .apply(lambda x: len(x.split()))
)

print(pd.Series(issues).describe())
