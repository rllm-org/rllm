import ast

from r2e.llms import LLMArgs, LLMCompletions
from r2e.pat.ast.explorer import build_ast, find_def_in_ast
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.repo_analysis.issues import random_issue_combination
from r2egym.repo_analysis.parse_pytest import parse_pytest_output
from r2egym.repo_analysis.execution_result_analysis import ExecutionResult
from r2egym.repo_analysis.repo_analysis_args import RepoAnalysisTestExtractArgs


def extract_issue(model_output: str):
    if "[ISSUE]" in model_output:
        model_output = model_output.split("[ISSUE]")[1]
    return model_output.split("[/ISSUE]")[0].strip()


pull_classes_repos = [
    "coveragepy",
]


def extract_test_fn(execution_result: ExecutionResult):
    improved_fn_names = execution_result.is_good_exec()[1]
    improved_fn_names = [fn.split(".")[-1] for fn in improved_fn_names]

    relevant_test_code = "```python"

    for improved_fn_name in improved_fn_names[:3]:
        for test_file_code in execution_result.test_file_codes:
            if improved_fn_name in test_file_code:
                ## TODO: slice the function from the test file
                improved_fn_ast = find_def_in_ast(
                    build_ast(test_file_code), improved_fn_name
                )
                try:
                    improved_fn_code = ast.unparse(improved_fn_ast)
                    if (
                        "self." in improved_fn_code
                        or execution_result.repo_name in pull_classes_repos
                    ):
                        ##  likely a class and uses parent (particularly in coveragepy)
                        if (
                            isinstance(improved_fn_ast, ast.FunctionDef)
                            and improved_fn_ast.parent is not None
                            and isinstance(improved_fn_ast.parent, ast.ClassDef)
                        ):
                            class_code = ast.unparse(improved_fn_ast.parent)
                            if len(class_code) < 20000:
                                improved_fn_code = (
                                    class_code
                                    + f"\n\n### THE CHANGED FUNCTION IS {improved_fn_ast.name} above\n"
                                )
                    if improved_fn_code not in relevant_test_code:
                        relevant_test_code += f"\n{improved_fn_code}\n"
                except Exception as e:
                    print(repr(e))

    relevant_test_code += "```"
    return relevant_test_code


def extract_test_fn_old_asserts(execution_result: ExecutionResult):
    improved_fn_names = execution_result.is_good_exec()[1]

    stdout_assert_parse_map = parse_pytest_output(
        execution_result.old_commit_res_stdout
    )
    relevant_test_code = "```pytest"

    for improved_fn_name in improved_fn_names[:3]:
        relevant_test_code += (
            "\n\n" + "=" * 80 + "\n" + improved_fn_name + "\n" + "=" * 80 + "\n\n"
        )
        truncated_assert = stdout_assert_parse_map[improved_fn_name]
        if len(truncated_assert) > 8000:
            truncated_assert = (
                truncated_assert[:3500]
                + "\n\n...TRUNCATED...\n\n"
                + truncated_assert[-3500:]
            )
        relevant_test_code += truncated_assert
    relevant_test_code += "```"
    return relevant_test_code


ISSUE_INSTRUCTIONS = """
As you are trying to generate synthetic issues, you will follow these guidelines

1. Keep the issue concise and informative.
2. Describe the failing test, including the input that causes the failure, the nature of the failure, and the expected behavior. Do NOT mention test functions or files directly. Do NOT mention pytest, hypothesis, or other testing frameworks.
3. Do not reveal the solution to the problem in the issue. Only describe the bug and the expected behavior.
4. If there are multiple failing tests, focus on the most informative one or a subset that best describes the general nature of the failure.
5. Describe the expected output of the failing test:
   - For errors, describe the error message.
   - For failing tests, mention what is supposed to happen. If the expected output is large and complex, describe the difference between the current and expected output instead of directly copying it (as human might do). Do NOT use assert statment is issue text, you are not writing test cases. 
6. Write the issue as a human would, using simple language without excessive formatting.
7. Use concrete terms to describe the nature of the failure. Avoid vague terms like "specific output" or "certain data".
8. INCLUDE test code to describe the bug but keep it brief and relevant. Truncate or simplify tests longer than 5-6 lines.
9. Do not mention external files unless absolutely necessary.
10. Format code snippets using triple backticks (```).

Before drafting the issue, analyze the following 
- Identify and quote key parts of the commit details and test results.
- What is the main problem highlighted by the test results?
- What is the expected behavior?
- What is the actual behavior or error message?
- How can you describe the issue concisely while providing enough information for developers to understand and investigate?
- Envision yourself as a human stumbling upon this bug. Provide the bug report from that perspective. Focus on clarity and naturalness in your writing.

After your analysis, draft the GitHub issue enclosed in [ISSUE] [/ISSUE] tags. The issue should include:
1. A clear and concise title (choose the best one from your brainstormed list)
2. A description of the problem 
    2.1 ensure adding a detailed example buggy code with sufficient explaintation
    2.2 ensure the example buggy code is natural, it should resemble a unittest, it should not have assertions 
    2.3 add details about the test scaffolding if necessary
3. Expected behavior
4. Actual behavior or error message

IMPORTANT: Strictly follow the above guidelines and use the provided test execution results to write the issue. Draw inspiration from the examples provided and make sure to provide good concise and natural issues. Remember to write the issue as a human would, focusing on clarity and relevance. For naturalness, envi

"""


def get_prompt(
    commit: ParsedCommit,
    execution_result: ExecutionResult,
    issues: str | None = None,
) -> tuple[str, str]:
    """
    Builds a synthetic issue for the given commit and execution result.
    It provides LLM with
        - commit hash
        - commit message
        - commit patch (w/o test file)
        - truncated stdouts (last 1000 characters)
        - improved test function names
        - improved test function codes
        - improved test function old assertion failures

    NOTE: that for pandas or repos with parameterized tests we use only one parametrized instance (split on `[`)
    """

    if issues is None:
        issues = random_issue_combination()

    prompt = f"""You are an expert software engineer tasked with creating informative GitHub issues based on commit details and test results. These issues will be used to help junior developers and machine learning systems understand the motivation behind commits. Your goal is to create concise, clear, and realistic issues that highlight bugs without revealing solutions.
    
The commit hash is {commit.new_commit_hash}. 
The commit message is: {commit.commit_message}.

The commit patch is:
```diff
{commit.get_patch(test_file=False)}. 
```

Additionally, we can write the following tests to check the correctness of the commit:
```diff
{commit.get_patch(non_test_file=False)}.
```


These tests detect the difference between the old and new commit. Particularly,

following is the execution result on the old commit:
{execution_result.old_commit_res_stdout_truncated}

following is the execution result on the new commit:
{execution_result.new_commit_res_stdout_truncated}

More specifically, the following tests that failed in the old commit now pass in the new commit:
{execution_result.find_improved_tests_formatted()}

Full test functions:
{extract_test_fn(execution_result)}

Incorrect test function outputs (failing assertion / errors from before):
{extract_test_fn_old_asserts(execution_result)}


Example Issues:

{issues}

{ISSUE_INSTRUCTIONS}
"""
    return prompt


def build_syn_issue(
    commit: ParsedCommit,
    execution_result: ExecutionResult,
    args: RepoAnalysisTestExtractArgs,
    issues: str | None = None,
    do_llm: bool = False,
) -> tuple[str, str]:
    prompt = get_prompt(commit, execution_result, issues)
    if not do_llm:
        return prompt, "BLANK", "BLANK"
    payload = [
        {"role": "user", "content": prompt},
    ]

    model_output = LLMCompletions.get_llm_completions((args), [payload])[0][0]

    issue = extract_issue(model_output)
    return prompt, model_output, issue
