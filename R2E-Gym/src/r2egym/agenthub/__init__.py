"""
constants and config for different agents
"""

# supported repos
SUPPORTED_REPOS = [
    "sympy",
    "pandas",
    "pillow",
    "numpy",
    "tornado",
    "coveragepy",
    "aiohttp",
    "pyramid",
    "datalad",
    "scrapy",
    "orange3",
]

# hidden / excluded files: to be hidden from the agent
SKIP_FILES = [
    "run_tests.sh",
    "syn_issue.json",
    "expected_test_output.json",
    "execution_result.json",
    "parsed_commit.json",
    "modified_files.json",
    "modified_entities.json",
    "r2e_tests",
]

SKIP_FILES_NEW = [
    "run_tests.sh",
    "r2e_tests",
]

# # continue msg for agent run loop (in case on null action)
# CONTINUE_MSG = """Please continue working on the task on whatever approach you think is suitable.
# If you think you have solved the task, please first send your answer to user through message and then finish the interaction.
# IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
# IMPORTANT: each response must have both a reasoning and function call. Again each response must have a function call.
# """

CONTINUE_MSG = """
You forgot to use a function call in your response. 
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""

# timeout for bash commands
CMD_TIMEOUT = 120  # seconds: 5 minutes
