"""OracleHarness: run a task's reference ``solution/solve.sh`` in the sandbox.

rLLM-native equivalent of Harbor's ``OracleAgent``. No LLM calls — the
harness uploads the task's ``solution/`` directory into the sandbox the
:class:`rllm.runner.Runner` already built (from ``environment/Dockerfile``,
``environment/setup.sh``, and ``[rllm].setup_commands``), executes
``solution/solve.sh``, and lets the Runner's auto-detected
``ShellScriptEvaluator`` score the result via ``tests/test.sh``.

Use it as a smoke-test for a Harbor-style task: confirms the image builds,
setup runs, and the verifier reports a baseline reward. If the oracle
reward matches the task README's expected baseline, the task is wired up
correctly and an LLM-driven harness can be trusted on top of it.

Usage::

    rllm eval /path/to/tasks/dir --agent oracle [--sandbox-backend modal]
"""

from __future__ import annotations

import logging
import shlex

from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)


class OracleHarness(SandboxedAgentFlow):
    """Upload ``solution/`` and run ``solve.sh`` inside the task sandbox."""

    name = "oracle"
    sandbox_backend = "docker"
    max_concurrent = 4

    # Mirror Harbor's ``env_paths.solution_dir`` convention so anyone
    # comparing the two implementations doesn't trip on path drift.
    _SOLUTION_DIR = "/solution"

    def run(self, task: Task, config: AgentConfig) -> Episode:
        sandbox = self.sandbox
        if sandbox is None:
            raise RuntimeError("OracleHarness requires a sandbox; the Runner should set one before run().")

        solution_dir = task.task_dir / "solution"
        solve_path = solution_dir / "solve.sh"
        if not solve_path.exists():
            raise FileNotFoundError(f"Task {task.id!r}: missing reference solution at {solve_path}")

        sandbox.exec(f"mkdir -p {self._SOLUTION_DIR}", user="root")
        sandbox.upload_dir(str(solution_dir), self._SOLUTION_DIR)
        container_solve = f"{self._SOLUTION_DIR}/solve.sh"
        sandbox.exec(f"chmod +x {shlex.quote(container_solve)}", user="root")

        agent_user = task.metadata.get("agent_user")
        timeout = float(task.metadata.get("agent_timeout", 7200))

        # Honor ``[solution].env`` from task.toml — same shape Harbor's
        # OracleAgent reads via ``resolve_env_vars``.
        solution_env = (task.metadata.get("solution", {}) or {}).get("env", {}) or {}
        env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in solution_env.items())
        cmd = (env_prefix + " " if env_prefix else "") + container_solve

        try:
            output = sandbox.exec(cmd, timeout=timeout, user=agent_user)
        except Exception as exc:
            logger.exception("oracle solve.sh failed for task %s", task.id)
            output = f"[oracle error] {type(exc).__name__}: {exc}"

        step = Step(id="oracle-solve", input=str(task.instruction), output=output)
        trajectory = Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.id,
            steps=[step],
            output=output,
        )
        return Episode(id=config.session_uid, task=task.id, trajectories=[trajectory])
