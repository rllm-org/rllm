"""BashTool: execute shell commands inside a sandbox."""

from __future__ import annotations

from rllm.experimental.agents.sandboxed_agent import _safe_exec
from rllm.sdk.sandbox.protocol import Sandbox


class BashTool:
    """Execute a bash command in the sandbox with output truncation."""

    OUTPUT_LIMIT = 16_000

    schema = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the sandbox. Use this to explore code, edit files, and run tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }

    def execute(self, sandbox: Sandbox, command: str = "", **kwargs) -> str:
        output = _safe_exec(sandbox, command, timeout=120)
        if len(output) > self.OUTPUT_LIMIT:
            output = output[: self.OUTPUT_LIMIT] + f"\n... [truncated, {len(output)} chars total]"
        return output
