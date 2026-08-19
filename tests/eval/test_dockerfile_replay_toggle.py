"""Tests for the ``[environment].replay_dockerfile`` toggle in :mod:`rllm.eval._resolution`.

Terminal-bench / Harbor tasks boot a fully-built image, so replaying the
Dockerfile's RUN steps on top double-applies the build (e.g. a `git clone`
failing because the destination already exists from the built image).
``_should_replay_dockerfile`` lets those tasks opt out and boot as-is.
"""

from __future__ import annotations

from pathlib import Path

from rllm.eval._resolution import _should_replay_dockerfile
from rllm.tasks.loader import _load_task_from_dir
from rllm.types import Task


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
