"""Tests for :func:`rllm.eval._resolution._dockerfile_run_commands`.

Non-docker sandbox backends (modal, daytona) replay a task Dockerfile's
``RUN`` steps by extracting each step as a shell command and exec'ing it
via ``bash -c`` (see ``_replay_dockerfile``). A multi-line ``RUN`` with
``\\``-continuations must collapse into one valid shell command.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rllm.eval._resolution import _dockerfile_run_commands
from rllm.types import Task


def _task_with_dockerfile(tmp_path: Path, dockerfile_body: str) -> Task:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(dockerfile_body)
    return Task(id="t", instruction="", metadata={}, dataset_dir=tmp_path)


def test_multiline_run_continuation_collapses_to_one_valid_shell_command(tmp_path):
    task = _task_with_dockerfile(
        tmp_path,
        "FROM python:3.11-slim\nRUN apt-get update && apt-get install -y \\\n    python3-pip \\\n    && rm -rf /var/lib/apt/lists/*\n",
    )

    commands = _dockerfile_run_commands(task)

    assert len(commands) == 1
    assert "\n" not in commands[0]
    # The extracted command must parse as a single valid shell command
    # (bash -n only checks syntax, no execution).
    result = subprocess.run(["bash", "-n", "-c", commands[0]], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_single_line_run_step_is_unaffected(tmp_path):
    task = _task_with_dockerfile(tmp_path, "FROM python:3.11-slim\nRUN echo hello\n")

    commands = _dockerfile_run_commands(task)

    assert commands == ["echo hello"]
