CONTINUE_MSG = """
You forgot to use a function call in your response.
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""

CMD_TIMEOUT = 120  # seconds

SKIP_FILES_NEW = ["run_tests.sh", "r2e_tests"]

DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
