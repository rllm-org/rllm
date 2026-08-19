"""Tests for Dockerfile RUN-continuation joining in :mod:`rllm.eval._resolution`.

``_dockerfile_run_commands`` joins ``\\``-continuations into a single valid
shell command (so multi-line ``RUN`` steps don't become invalid ``bash``).
"""

from __future__ import annotations

from pathlib import Path

from rllm.eval._resolution import _dockerfile_run_commands
from rllm.types import Task


def _task_with_dockerfile(tmp_path: Path, dockerfile: str) -> Task:
    env = tmp_path / "environment"
    env.mkdir(parents=True, exist_ok=True)
    (env / "Dockerfile").write_text(dockerfile)
    return Task(id="t", instruction="", metadata={}, dataset_dir=tmp_path)


def test_multiline_run_joins_with_space_not_newline(tmp_path):
    task = _task_with_dockerfile(
        tmp_path,
        "FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y \\\n    python3-pip \\\n    && rm -rf /var/lib/apt/lists/*\n",
    )
    cmds = _dockerfile_run_commands(task)
    assert cmds == ["apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*"]
    # The old bug joined with "\n", which bash rejects ("syntax error near '&&'").
    assert "\n" not in cmds[0]


def test_copy_directives_are_skipped(tmp_path):
    task = _task_with_dockerfile(
        tmp_path,
        "FROM ubuntu:24.04\nCOPY input_files/data.json /app/data.json\nRUN echo hi\nADD x.tar /app/\n",
    )
    assert _dockerfile_run_commands(task) == ["echo hi"]


def test_single_line_run_unchanged(tmp_path):
    task = _task_with_dockerfile(tmp_path, "FROM ubuntu:24.04\nRUN pip3 install ansible-core==2.16.3\n")
    assert _dockerfile_run_commands(task) == ["pip3 install ansible-core==2.16.3"]
