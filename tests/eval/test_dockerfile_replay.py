"""Tests for Dockerfile RUN-replay handling in :mod:`rllm.eval._resolution`.

Covers two fixes:

* ``_dockerfile_run_commands`` joins ``\\``-continuations into a single valid
  shell command (so multi-line ``RUN`` steps don't become invalid ``bash``).
* ``_should_replay_dockerfile`` honors ``[environment].replay_dockerfile``:
  default ``True`` (SWE-bench: replay RUN steps on a base image), ``False`` for
  fully-built terminal-bench/Harbor images (boot as-is, no double-apply).
"""

from __future__ import annotations

from pathlib import Path

from rllm.eval._resolution import _dockerfile_run_commands, _should_replay_dockerfile
from rllm.tasks.loader import _load_task_from_dir
from rllm.types import Task


def _task_with_dockerfile(tmp_path: Path, dockerfile: str, metadata: dict | None = None) -> Task:
    env = tmp_path / "environment"
    env.mkdir(parents=True, exist_ok=True)
    (env / "Dockerfile").write_text(dockerfile)
    return Task(id="t", instruction="", metadata=metadata or {}, dataset_dir=tmp_path)


# ---------------------------------------------------------------------------
# _dockerfile_run_commands: continuation join + COPY skipping
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _should_replay_dockerfile: the [environment].replay_dockerfile toggle
# ---------------------------------------------------------------------------


def test_replay_defaults_true_when_unset():
    assert _should_replay_dockerfile(Task(id="t", instruction="", metadata={}, dataset_dir=Path("."))) is True
    task = Task(id="t", instruction="", metadata={"environment": {"docker_image": "base"}}, dataset_dir=Path("."))
    assert _should_replay_dockerfile(task) is True


def test_replay_disabled_when_flag_false():
    task = Task(
        id="t",
        instruction="",
        metadata={"environment": {"docker_image": "prebuilt", "replay_dockerfile": False}},
        dataset_dir=Path("."),
    )
    assert _should_replay_dockerfile(task) is False


def test_replay_flag_read_from_task_toml(tmp_path):
    """End-to-end: [environment].replay_dockerfile in task.toml reaches metadata."""
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN echo hi\n")
    (tmp_path / "task.toml").write_text('[environment]\ndocker_image = "org/img:t"\nreplay_dockerfile = false\n')
    task = _load_task_from_dir(tmp_path, dataset_dir=tmp_path)
    assert _should_replay_dockerfile(task) is False
