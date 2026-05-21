from __future__ import annotations

import pytest
from swe.tasks.common import (
    PatchApplyError,
    apply_patch_file,
    grade_in_fresh_env,
    reinitialize_git_repo,
    run_sandbox,
    write_file_b64,
)


class FakeEnv:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.commands = []
        self.closed = False

    def execute(self, payload, *, cwd, timeout):
        self.commands.append((payload["command"], cwd, timeout))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return {"returncode": 0, "output": ""}

    def close(self):
        self.closed = True


def test_run_sandbox_raises_with_command_context_on_nonzero():
    env = FakeEnv([{"returncode": 2, "output": "bad output"}])

    with pytest.raises(RuntimeError, match="git status"):
        run_sandbox(env, "git status", cwd="/repo", timeout=5)


def test_write_file_b64_uses_quoted_base64_chunks():
    env = FakeEnv()

    write_file_b64(env, path="/tmp/value.txt", content="hello\nworld", cwd="/repo")

    assert env.commands[0][0] == "rm -f /tmp/value.txt"
    assert "base64 -d >> /tmp/value.txt" in env.commands[1][0]


def test_apply_patch_file_falls_back_to_three_way():
    env = FakeEnv([
        {"returncode": 1, "output": "clean failed"},
        {"returncode": 0, "output": "three-way ok"},
    ])

    apply_patch_file(env, patch_path="/tmp/p.diff", cwd="/repo", short_id="abc", label="agent patch")

    assert env.commands[0][0] == "git apply -v /tmp/p.diff"
    assert "--3way" in env.commands[1][0]


def test_apply_patch_file_raises_patch_apply_error_after_failed_fallback():
    env = FakeEnv([
        {"returncode": 1, "output": "clean failed"},
        {"returncode": 1, "output": "three-way failed"},
    ])

    with pytest.raises(PatchApplyError, match="agent patch apply failed"):
        apply_patch_file(env, patch_path="/tmp/p.diff", cwd="/repo", short_id="abc", label="agent patch")


def test_reinitialize_git_repo_hides_history_with_agent_start_commit():
    env = FakeEnv()

    reinitialize_git_repo(env, cwd="/repo", user_email="x@y", user_name="tester")

    commands = [command for command, _cwd, _timeout in env.commands]
    assert commands == [
        "rm -rf .git",
        "git init",
        "git config user.email x@y",
        "git config user.name tester",
        "git config commit.gpgsign false",
        "git add -A",
        "git commit --allow-empty -m agent-start",
    ]


def test_grade_in_fresh_env_closes_on_success_and_failure():
    env = FakeEnv()

    result = grade_in_fresh_env(lambda _task: env, {"instance_id": "x"}, lambda _task, _env: {"reward": 1.0})

    assert result == {"reward": 1.0}
    assert env.closed

    failing_env = FakeEnv()

    def fail(_task, _env):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        grade_in_fresh_env(lambda _task: failing_env, {"instance_id": "x"}, fail)
    assert failing_env.closed
