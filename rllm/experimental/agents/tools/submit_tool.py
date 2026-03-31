"""SubmitTool: signal task completion and capture the answer."""

from __future__ import annotations

from rllm.experimental.agents.sandboxed_agent import _safe_exec
from rllm.sdk.sandbox.protocol import Sandbox


class SubmitTool:
    """Signal that the agent has finished and capture the final answer."""

    schema = {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your final answer. Call this when you are done with the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your final answer or summary of changes made.",
                    }
                },
                "required": [],
            },
        },
    }

    def execute(self, sandbox: Sandbox, answer: str = "", **kwargs) -> str:
        # Capture git diff as the canonical output
        diff = _safe_exec(sandbox, "git diff 2>/dev/null || echo 'No git repo'")
        if answer:
            return f"Answer submitted: {answer}\n\nGit diff:\n{diff}"
        return f"Submitted.\n\nGit diff:\n{diff}"
