"""TaskExecutor: sandboxed agent that runs on a task directory.

Sets up the sandbox from a task's ``task.toml`` configuration, uploads
environment files, and runs the agent in the sandbox.

Lifecycle (managed by EvalRunner):
1. ``setup_sandbox()`` — create sandbox, upload env files, run setup
2. ``run()`` — present instruction to LLM, agent interacts with sandbox
3. ``teardown_sandbox()`` — cleanup (inherited from SandboxedAgentFlow)
"""

from __future__ import annotations

import logging
import uuid

from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow, _safe_exec, create_sandbox
from rllm.experimental.eval.types import AgentConfig, Task
from rllm.tasks.task_config import LoadedTask, load_task
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)


class TaskExecutor(SandboxedAgentFlow):
    """Agent flow that executes a task directory inside a sandbox.

    Reads ``task.toml`` to configure the sandbox (image, workdir, env vars),
    uploads ``environment/files/`` into the sandbox, and runs setup commands.

    The ``run()`` method presents the task instruction to the LLM and lets
    it interact with the sandbox via tool calls.
    """

    def setup_sandbox(self, task: dict, config: AgentConfig) -> None:
        """Create and configure sandbox from the task directory."""
        loaded = load_task(task["task_path"])
        self._loaded_task = loaded

        # Check sandbox compatibility
        required = loaded.rllm.sandbox
        if required != "any" and required != self.sandbox_backend:
            raise ValueError(f"Task '{loaded.task_name}' requires sandbox={required!r}, but got --sandbox-backend={self.sandbox_backend!r}")

        # Create sandbox
        image = loaded.image
        task_id = task.get("task_id", loaded.task_name)
        # Sanitize for Docker container naming: only [a-zA-Z0-9_.-] allowed
        import re

        task_id_safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", task_id)
        name = f"rllm-{task_id_safe}-{uuid.uuid4().hex[:6]}"
        self._sandbox = create_sandbox(self.sandbox_backend, name=name, image=image)

        # Create workdir
        _safe_exec(self._sandbox, f"mkdir -p {loaded.workdir}", timeout=30)

        # Upload environment/files/ into workdir
        files_dir = loaded.path / "environment" / "files"
        if files_dir.is_dir():
            self._sandbox.upload_dir(str(files_dir), loaded.workdir)

        # Run environment/setup.sh if present
        setup_script = loaded.path / "environment" / "setup.sh"
        if setup_script.exists():
            self._sandbox.upload_file(str(setup_script), "/tmp/rllm_setup.sh")
            _safe_exec(self._sandbox, "chmod +x /tmp/rllm_setup.sh && /tmp/rllm_setup.sh", timeout=300)

        # Run [rllm] setup_commands
        for cmd in loaded.rllm.setup_commands:
            _safe_exec(self._sandbox, cmd, timeout=300)

        # Set environment variables from [environment].env
        if loaded.env_vars:
            exports = " && ".join(f"export {k}='{v}'" for k, v in loaded.env_vars.items())
            _safe_exec(self._sandbox, exports, timeout=10)

    def run(self, task: Task, config: AgentConfig) -> Episode:
        """Run the agent on the task.

        Default implementation: sends instruction to the LLM and collects
        the response as a single-turn trajectory.  Subclasses or custom
        agents can override for multi-turn tool-use loops.
        """
        from openai import OpenAI

        instruction = task.data.get("instruction", "")
        loaded: LoadedTask = getattr(self, "_loaded_task", None) or load_task(task.data["task_path"])

        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        messages = [
            {"role": "system", "content": "You are a skilled software engineer. Complete the task by executing commands in the sandbox."},
            {"role": "user", "content": instruction},
        ]

        max_turns = loaded.rllm.max_turns or 50
        steps: list[Step] = []

        for turn in range(max_turns):
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
            )

            assistant_msg = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_msg})

            steps.append(
                Step(
                    id=f"step-{turn}",
                    input=messages[-2]["content"] if turn == 0 else "",
                    output=assistant_msg,
                )
            )

            # Check if agent signals completion
            if _is_done(assistant_msg):
                break

            # If sandbox is available, the agent can execute commands
            # by wrapping them in ```bash blocks
            command = _extract_command(assistant_msg)
            if command and self._sandbox is not None:
                try:
                    result = self._sandbox.exec(command, timeout=float(loaded.agent_timeout))
                except Exception as e:
                    result = f"Error: {e}"

                messages.append({"role": "user", "content": f"Command output:\n{result}"})
            else:
                # No command to execute — agent is done
                break

        trajectory = Trajectory(
            uid=config.session_uid,
            name="task-executor",
            task=task.data.get("task_id", ""),
            steps=steps,
            output=steps[-1].output if steps else "",
        )

        return Episode(
            id=config.session_uid,
            task=task.data.get("task_id", ""),
            trajectories=[trajectory],
        )

    def get_image(self, task: dict) -> str:
        """Return container image from the task's task.toml."""
        try:
            loaded = load_task(task["task_path"])
            return loaded.image
        except Exception:
            return self.image


def _is_done(text: str) -> bool:
    """Heuristic: check if the agent signals it's finished."""
    lower = text.lower().strip()
    done_signals = ["task completed", "task is complete", "done", "finished", "i have completed"]
    return any(lower.endswith(sig) or lower.startswith(sig) for sig in done_signals)


def _extract_command(text: str) -> str | None:
    """Extract a bash command from markdown code fences."""
    import re

    match = re.search(r"```(?:bash|shell|sh)\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
