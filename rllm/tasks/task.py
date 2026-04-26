"""Task: an RL environment with an evaluator.

A Task is loaded from a directory and knows:
  - How to materialize itself in a sandbox (``setup``)
  - How to score an attempt (``evaluate``)
  - What instruction the agent receives (``instruction``)
  - What container image to use (``get_image``)

A Task does NOT know:
  - Which agent will solve it
  - What model the agent uses
  - How the agent loops / extracts commands

Those concerns live in :class:`AgentHarness`.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.sdk.sandbox.protocol import Sandbox

logger = logging.getLogger(__name__)


# Reward file search order (first existing file wins)
_REWARD_PATHS = [
    "/tmp/rllm/reward.json",
    "/logs/verifier/reward.json",
    "/logs/verifier/reward.txt",
]


@dataclass
class RllmExtensions:
    """The ``[rllm]`` section in task.toml — rLLM-specific overrides."""

    sandbox: str = "docker"
    """Backend compatibility: ``"any"`` | ``"docker"`` | ``"local"`` | ``"modal"`` | ``"daytona"``."""

    setup_commands: list[str] = field(default_factory=list)
    """Lightweight environment setup commands (run after Dockerfile/files, before agent)."""

    max_turns: int | None = None
    """Optional cap on agent interaction turns."""

    reward_file: str | None = None
    """Override reward file location (default: check Harbor paths then ``/tmp/rllm/reward.json``)."""


@dataclass
class Task:
    """An RL task: environment + instruction + evaluator.

    Loaded from a directory containing:
      - ``task.toml``         — Harbor-compatible config + ``[rllm]`` section
      - ``instruction.md``    — what the agent should do
      - ``environment/``      — ``Dockerfile`` and/or ``files/`` and ``setup.sh``
      - ``tests/``            — verifier scripts (``test.sh``)
    """

    path: Path
    config: dict[str, Any]
    rllm: RllmExtensions
    instruction: str

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, task_dir: str | Path) -> Task:
        """Load a task directory."""
        task_dir = Path(task_dir).resolve()

        config_path = task_dir / "task.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"task.toml not found in {task_dir}")

        raw = tomllib.loads(config_path.read_text())
        rllm_section = raw.pop("rllm", {})
        known_fields = set(RllmExtensions.__dataclass_fields__.keys())
        rllm = RllmExtensions(**{k: v for k, v in rllm_section.items() if k in known_fields})

        instruction_path = task_dir / "instruction.md"
        instruction = instruction_path.read_text() if instruction_path.exists() else ""

        return cls(path=task_dir, config=raw, rllm=rllm, instruction=instruction)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.config.get("task", {}).get("name", self.path.name)

    @property
    def workdir(self) -> str:
        return self.config.get("environment", {}).get("workdir", "/workspace")

    @property
    def env_vars(self) -> dict[str, str]:
        return self.config.get("environment", {}).get("env", {})

    @property
    def verifier_timeout(self) -> int:
        return self.config.get("verifier", {}).get("timeout_sec", 600)

    @property
    def agent_timeout(self) -> int:
        return self.config.get("agent", {}).get("timeout_sec", 600)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.config.get("metadata", {})

    def required_sandbox_backend(self) -> str:
        """Return the sandbox compatibility hint from ``[rllm].sandbox``."""
        return self.rllm.sandbox

    # ------------------------------------------------------------------
    # Image resolution
    # ------------------------------------------------------------------

    def get_image(self, sandbox_backend: str) -> str:
        """Resolve the container image for this task.

        Priority:
          1. If ``environment/Dockerfile`` exists and backend is ``docker``:
             build from the Dockerfile.
          2. Otherwise: use ``[environment].docker_image`` from task.toml.
          3. Default: ``python:3.11-slim``.
        """
        configured = self.config.get("environment", {}).get("docker_image", "python:3.11-slim")

        dockerfile = self.path / "environment" / "Dockerfile"
        if dockerfile.exists() and sandbox_backend == "docker":
            return _build_docker_image(self.path / "environment", self.name)

        return configured

    # ------------------------------------------------------------------
    # Sandbox preparation
    # ------------------------------------------------------------------

    def setup(self, sandbox: Sandbox) -> None:
        """Prepare the sandbox: upload files, run setup, set env vars."""
        # Workdir
        _safe_exec(sandbox, f"mkdir -p {self.workdir}", timeout=30)

        # environment/files/ → workdir
        files_dir = self.path / "environment" / "files"
        if files_dir.is_dir():
            sandbox.upload_dir(str(files_dir), self.workdir)

        # environment/setup.sh
        setup_script = self.path / "environment" / "setup.sh"
        if setup_script.exists():
            sandbox.upload_file(str(setup_script), "/tmp/rllm_setup.sh")
            _safe_exec(sandbox, "chmod +x /tmp/rllm_setup.sh && /tmp/rllm_setup.sh", timeout=300)

        # [rllm].setup_commands
        for cmd in self.rllm.setup_commands:
            _safe_exec(sandbox, cmd, timeout=300)

        # [environment].env
        if self.env_vars:
            exports = " && ".join(f"export {k}='{v}'" for k, v in self.env_vars.items())
            _safe_exec(sandbox, exports, timeout=10)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, sandbox: Sandbox) -> EvalOutput:
        """Run the verifier in the sandbox and return reward."""
        tests_dir = self.path / "tests"
        if not tests_dir.is_dir():
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": f"no tests/ directory in {self.path}"})

        # Prepare reward directories
        try:
            sandbox.exec("mkdir -p /tmp/rllm /logs/verifier", timeout=10)
        except Exception:
            pass

        # Upload to /tests/ (Harbor convention — scripts may reference /tests/*.py)
        sandbox.upload_dir(str(tests_dir), "/tests")

        # Find and run the test script
        test_script = _find_test_script(tests_dir)
        if test_script is None:
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no test.sh or test.py found in tests/"})

        try:
            sandbox.exec(
                f"chmod +x /tests/{test_script} && cd {self.workdir} && /tests/{test_script}",
                timeout=float(self.verifier_timeout),
            )
        except Exception as e:
            logger.warning("Test script execution error for %s: %s", self.name, e)

        # Read reward
        reward_paths = list(_REWARD_PATHS)
        if self.rllm.reward_file:
            reward_paths.insert(0, self.rllm.reward_file)
        return _read_reward_from_sandbox(sandbox, reward_paths)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_exec(sandbox: Sandbox, command: str, timeout: float | None = None) -> str:
    """Execute command, log on failure but don't raise."""
    try:
        return sandbox.exec(command, timeout=timeout)
    except Exception as e:
        logger.debug("Command failed (suppressed): %s — %s", command[:200], e)
        return ""


def _find_test_script(tests_dir: Path) -> str | None:
    """Find the test script name (relative to tests_dir)."""
    for name in ("test.sh", "test.py", "test.bat"):
        if (tests_dir / name).exists():
            return name
    return None


def _read_reward_from_sandbox(sandbox: Sandbox, paths: list[str]) -> EvalOutput:
    """Try reading reward from the sandbox at each path in order."""
    for path in paths:
        try:
            check = sandbox.exec(f"test -f {path} && echo yes || echo no", timeout=10).strip()
            if check != "yes":
                continue
            raw = sandbox.exec(f"cat {path}", timeout=10).strip()
            if not raw:
                continue
            if path.endswith(".txt"):
                reward = float(raw)
                return EvalOutput(reward=reward, is_correct=reward >= 1.0)
            return _parse_reward_json(raw)
        except Exception as e:
            logger.debug("Could not read reward from %s: %s", path, e)
            continue

    logger.warning("No reward file found at any of: %s", paths)
    return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no reward file found"})


def _parse_reward_json(raw: str) -> EvalOutput:
    """Parse a JSON reward file into an EvalOutput.

    Supports both ``{"reward": 0.5}`` and Harbor-style ``{"rewards": {...}}``.
    """
    data = json.loads(raw)

    if "reward" in data:
        reward = float(data["reward"])
    elif "rewards" in data and data["rewards"]:
        reward = sum(float(v) for v in data["rewards"].values()) / len(data["rewards"])
    else:
        reward = 0.0

    is_correct = data.get("is_correct", reward >= 1.0)

    signals: list[Signal] = []
    for key, val in data.get("signals", {}).items():
        signals.append(Signal(name=key, value=float(val)))
    for key, val in data.get("rewards", {}).items():
        if key != "reward":
            signals.append(Signal(name=key, value=float(val)))

    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=signals,
        metadata=data.get("metadata", {}),
    )


def _build_docker_image(context_dir: Path, task_name: str) -> str:
    """Build a Docker image from a Dockerfile via subprocess (avoids Python SDK credential helpers)."""
    tag = "rllm-task-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", task_name).lower()
    logger.info("Building Docker image '%s' from %s", tag, context_dir)
    result = subprocess.run(
        ["docker", "build", "-t", tag, "--rm", "."],
        cwd=str(context_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed for {task_name}:\n{result.stderr[:1000]}")
    return tag
