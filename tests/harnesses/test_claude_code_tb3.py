"""TB3 task configuration passed to the native Claude Code harness."""

from pathlib import Path

from rllm.harnesses.claude_code import ClaudeCodeHarness
from rllm.types import AgentConfig, Task


class _Sandbox:
    def __init__(self):
        self.commands: list[str] = []

    def exec(self, command, timeout=None, user=None):  # noqa: ARG002
        self.commands.append(command)
        return ""


def test_task_mcp_and_skills_are_registered_for_claude_code():
    task = Task(
        id="tb3",
        instruction="",
        dataset_dir=Path("."),
        metadata={
            "environment": {
                "skills_dir": "/app/.agents/skills",
                "mcp_servers": [
                    {
                        "name": "playwright",
                        "transport": "sse",
                        "url": "http://playwright-mcp:3080/sse",
                    }
                ],
            }
        },
    )
    sandbox = _Sandbox()
    ClaudeCodeHarness().write_configs(
        sandbox,
        task,
        AgentConfig(base_url="http://gateway/v1", model="model", session_uid="s"),
        {},
    )
    command = sandbox.commands[0]
    assert "/app/.agents/skills/." in command
    assert '"mcpServers"' in command
    assert '"playwright"' in command
    assert '"type": "sse"' in command


def _invocation(harness: ClaudeCodeHarness) -> str:
    task = Task(id="tb3", instruction="do the thing", dataset_dir=Path("."), metadata={})
    return harness.build_invocation("do the thing", task, AgentConfig(base_url="http://x/v1", model="m", session_uid="s"))


def test_reasoning_effort_omitted_by_default():
    # Matches both the CLI's own default and harbor's (its CliFlag has no default).
    assert "--effort" not in _invocation(ClaudeCodeHarness())


def test_reasoning_effort_is_passed_as_cli_flag():
    # The leaderboard config (`--ak reasoning_effort=max`) is a harness flag, not
    # a sampling param — it must land on the invocation, never in the request body.
    harness = ClaudeCodeHarness()
    assert harness.configure({"reasoning_effort": "max"}) == {}
    assert "--effort max " in _invocation(harness)


def test_reasoning_effort_rejects_unknown_level():
    harness = ClaudeCodeHarness()
    try:
        harness.configure({"reasoning_effort": "turbo"})
    except ValueError as e:
        assert "turbo" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown effort level")
