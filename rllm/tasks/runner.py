"""TaskRunner: orchestrates Task + AgentHarness inside a sandbox.

The runner implements :class:`AgentFlow` so it plugs into the existing
``EvalRunner`` and training engine without changes.  The harness is the
swappable bit — same task, different agent.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow, create_sandbox
from rllm.tasks.task import Task

if TYPE_CHECKING:
    from rllm.eval.types import AgentConfig
    from rllm.eval.types import Task as EvalTask
    from rllm.tasks.harness import AgentHarness
    from rllm.types import Episode

logger = logging.getLogger(__name__)


class TaskRunner(SandboxedAgentFlow):
    """Implements ``AgentFlow`` for sandboxed tasks.

    Lifecycle (driven by ``EvalRunner``):

    1. :meth:`setup_sandbox` — load Task, create sandbox, ``task.setup`` then ``harness.setup``
    2. :meth:`run` — ``harness.run(task, sandbox, config)`` → ``Episode``
    3. :meth:`teardown_sandbox` — ``sandbox.close()`` (inherited)

    The harness is selected at construction time via ``--agent <name>`` in the CLI.
    """

    def __init__(self, harness: AgentHarness, sandbox_backend: str = "docker", max_concurrent: int = 4):
        super().__init__()
        self.harness = harness
        self.sandbox_backend = sandbox_backend
        self.max_concurrent = max_concurrent
        self._task: Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_sandbox(self, task: dict, config: AgentConfig) -> None:
        """Create sandbox, prepare task, set up harness."""
        loaded = Task.load(task["task_path"])
        self._task = loaded

        # Sandbox compatibility check
        required = loaded.required_sandbox_backend()
        if required != "any" and required != self.sandbox_backend:
            raise ValueError(f"Task '{loaded.name}' requires sandbox={required!r}, but got --sandbox-backend={self.sandbox_backend!r}")

        # Resolve image (builds Dockerfile if present)
        image = loaded.get_image(self.sandbox_backend)

        # Sanitize for Docker container naming
        task_id = task.get("task_id", loaded.name)
        task_id_safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", task_id)
        sandbox_name = f"rllm-{task_id_safe}-{uuid.uuid4().hex[:6]}"

        self._sandbox = create_sandbox(self.sandbox_backend, name=sandbox_name, image=image)

        # Task-specific setup (upload files, run setup commands)
        loaded.setup(self._sandbox)

        # Harness-specific setup (install agent tools, configure env)
        self.harness.setup(self._sandbox, config)

    def run(self, task: EvalTask, config: AgentConfig) -> Episode:
        """Drive the harness, wrap the resulting Trajectory in an Episode."""
        from rllm.types import Episode

        if self._task is None or self._sandbox is None:
            raise RuntimeError("setup_sandbox() must be called before run()")

        trajectory = self.harness.run(self._task, self._sandbox, config)

        return Episode(
            id=config.session_uid,
            task=self._task.name,
            trajectories=[trajectory],
        )

    # ------------------------------------------------------------------
    # Image hint (used by SandboxedAgentFlow.setup_sandbox in some paths)
    # ------------------------------------------------------------------

    def get_image(self, task: dict) -> str:
        """Return image for this task (without rebuilding if Dockerfile)."""
        try:
            loaded = Task.load(task["task_path"])
            return loaded.config.get("environment", {}).get("docker_image", self.image)
        except Exception:
            return self.image
