"""Format sandbox command output for the agent's conversation context."""

_TRUNCATION_LINES = 40


def format_observation(output: str, error_code: str, action_name: str) -> str:
    """Format command output into an observation string.

    For bash commands: includes exit code and truncates long output
    (keeps first/last 40 lines) to save LLM context.
    For tool commands (file_editor, search, etc.): light header only,
    since tool scripts handle their own truncation internally.
    """
    if action_name in ("execute_bash", "bash"):
        lines = output.splitlines() if output else []
        if len(lines) > 2 * _TRUNCATION_LINES:
            top = "\n".join(lines[:_TRUNCATION_LINES])
            bottom = "\n".join(lines[-_TRUNCATION_LINES:])
            divider = "-" * 50
            output = (
                f"{top}\n"
                f"{divider}\n"
                f"<Observation truncated in middle for saving context>\n"
                f"{divider}\n"
                f"{bottom}"
            )
        return (
            f"Exit code: {error_code}\n"
            f"Execution output of [{action_name}]:\n"
            f"{output}"
        )

    return f"Execution output of [{action_name}]:\n{output}"
