"""Task configuration: load a task directory's task.toml + instruction.md.

Uses only stdlib (tomllib, dataclasses, pathlib) — no Harbor dependency.
The task.toml format is Harbor-compatible; rLLM extensions live under [rllm].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


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
class LoadedTask:
    """Everything rLLM needs from a task directory.

    ``raw_config`` preserves the full task.toml as a dict so that
    Harbor-native fields are available without importing Harbor models.
    """

    path: Path
    raw_config: dict[str, Any]
    rllm: RllmExtensions
    instruction: str

    # -- Convenience accessors into raw_config --

    @property
    def task_name(self) -> str:
        return self.raw_config.get("task", {}).get("name", self.path.name)

    @property
    def image(self) -> str:
        return self.raw_config.get("environment", {}).get("docker_image", "python:3.11-slim")

    @property
    def workdir(self) -> str:
        return self.raw_config.get("environment", {}).get("workdir", "/workspace")

    @property
    def env_vars(self) -> dict[str, str]:
        return self.raw_config.get("environment", {}).get("env", {})

    @property
    def verifier_timeout(self) -> int:
        return self.raw_config.get("verifier", {}).get("timeout_sec", 600)

    @property
    def agent_timeout(self) -> int:
        return self.raw_config.get("agent", {}).get("timeout_sec", 600)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.raw_config.get("metadata", {})


def load_task(task_dir: str | Path) -> LoadedTask:
    """Load a task directory.  Only stdlib — no Harbor dependency.

    Args:
        task_dir: Path to a directory containing ``task.toml`` and ``instruction.md``.

    Returns:
        A ``LoadedTask`` with the parsed config and instruction text.

    Raises:
        FileNotFoundError: If ``task.toml`` or ``instruction.md`` is missing.
    """
    task_dir = Path(task_dir).resolve()

    config_path = task_dir / "task.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"task.toml not found in {task_dir}")

    raw = tomllib.loads(config_path.read_text())

    # Pop [rllm] section — everything else is Harbor-native
    rllm_section = raw.pop("rllm", {})
    known_fields = {f.name for f in RllmExtensions.__dataclass_fields__.values()}
    rllm = RllmExtensions(**{k: v for k, v in rllm_section.items() if k in known_fields})

    # Read instruction
    instruction_path = task_dir / "instruction.md"
    instruction = instruction_path.read_text() if instruction_path.exists() else ""

    return LoadedTask(
        path=task_dir,
        raw_config=raw,
        rllm=rllm,
        instruction=instruction,
    )
